from __future__ import print_function
from __future__ import division
import tensorflow as tf
from src.utils import tf_utils


class BiLSTMChar(object):
    """
    A bidirectional LSTM for embedding tokens.
    """

    def __init__(self, char_domain_size, char_embedding_dim, hidden_dim, embeddings=None):
        """
        Initializing a Bi-LSTM layer for character embedding of a word
        :param char_domain_size:
        :param char_embedding_dim:
        :param hidden_dim:
        :param embeddings:
        """
        ################################################################################
        # Initializing the object variables
        ################################################################################
        # Domain size if the characters : NUmber of unique characters
        self.char_domain_size = char_domain_size
        # Size of the embedding
        self.embedding_size = char_embedding_dim
        # Size of the hidden dimension
        self.hidden_dim = hidden_dim
        # Size of the output: double of the hidden dimension
        self.output_size = 2 * self.hidden_dim

        print("Bi-LSTM char embedding model")
        print("embedding dim: ", self.embedding_size)
        print("out dim: ", self.output_size)

        ################################################################################
        # Initializing the TensorFlow variables
        ################################################################################

        # PLACEHOLDERS
        # char embedding input
        self.input_chars = tf.placeholder(tf.int64, [None, None], name="input_chars")
        # batch size
        self.batch_size = tf.placeholder(tf.int32, None, name="batch_size")
        # Max number of words in a sentence
        self.max_seq_len = tf.placeholder(tf.int32, None, name="max_seq_len")
        # Max length of a token
        self.max_tok_len = tf.placeholder(tf.int32, None, name="max_tok_len")
        # dropout probabilities
        self.input_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="input_dropout_keep_prob")
        # sequence lengths for each sentence
        self.sequence_lengths = tf.placeholder(tf.int32, [None, None], name="sequence_lengths")
        # token length of all tokens in a sentence
        self.token_lengths = tf.placeholder(tf.int32, [None, None], name="tok_lengths")
        # Embedding layer
        shape = (char_domain_size - 1, self.embedding_size)
        self.char_embeddings = tf_utils.initialize_embeddings(shape, name="char_embeddings", pretrained=embeddings)

        ################################################################################
        # Processing the forward layer
        ################################################################################
        self.outputs = self.forward(self.input_chars, self.input_dropout_keep_prob, reuse=False)

    def forward(self, input_chars, input_dropout_keep_prob, reuse=True):
        """
        Computing  the forward pass of the network
        :param input_chars:
        :param input_dropout_keep_prob:
        :param reuse:
        :return:
        """
        with tf.variable_scope("char-forward", reuse=reuse):
            char_embeddings_lookup = tf.nn.embedding_lookup(self.char_embeddings, input_chars)
            char_embeddings_flat = tf.reshape(char_embeddings_lookup, tf.stack(
                [self.batch_size * self.max_seq_len, self.max_tok_len, self.embedding_size]))
            tok_lens_flat = tf.reshape(self.token_lengths, [self.batch_size * self.max_seq_len])
            input_feats_drop = tf.nn.dropout(char_embeddings_flat, input_dropout_keep_prob)
            with tf.name_scope("char-bilstm"):
                fwd_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
                bwd_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
                lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell, cell_bw=bwd_cell, dtype=tf.float32,
                                                                  inputs=input_feats_drop,
                                                                  parallel_iterations=32, swap_memory=False,
                                                                  sequence_length=tok_lens_flat)
                outputs_fw = lstm_outputs[0]
                outputs_bw = lstm_outputs[1]
                # this is batch*output_size (flat)
                fw_output = tf_utils.last_relevant(outputs_fw, tok_lens_flat)
                # this is batch * max_seq_len * output_size
                bw_output = outputs_bw[:, 0, :]
                hidden_outputs = tf.concat(axis=1, values=[fw_output, bw_output])
                hidden_outputs_unflat = tf.reshape(hidden_outputs,
                                                   tf.stack([self.batch_size, self.max_seq_len, self.output_size]))
        return hidden_outputs_unflat


class BiLSTM(object):
    """
    A bidirectional LSTM for text classification.
    This class implements a BiLSTM layer as described in the paper : https://arxiv.org/pdf/1508.01991.pdf
    """

    def __init__(self, num_classes, vocab_size, shape_domain_size, char_domain_size, char_size,
                 embedding_size, shape_size, nonlinearity, viterbi, hidden_dim, char_embeddings, embeddings=None):

        """
        Initializing a BiLSTM layer with parameters
        :param num_classes:
        :param vocab_size:
        :param char_domain_size:
        :param char_size:
        :param embedding_size:
        :param shape_size:
        :param nonlinearity:
        :param viterbi:
        :param hidden_dim:
        :param char_embeddings:
        :param embeddings:
        """
        ################################################################################
        # Initializing the object variables
        ################################################################################
        # Number of output classes : In our case number of distinct labels
        self.num_classes = num_classes
        # Maximum character size in a word token : Used for padding and boolean masking
        self.char_size = char_size
        # Character embeddings of the word
        self.char_embeddings = char_embeddings
        # Dimension of word embedding : Useful when there is no pre-trained word embedding
        self.embedding_size = embedding_size
        # Dimension of the hidden layer of the LSTM
        self.hidden_dim = hidden_dim
        # Type of non-linear function used
        self.nonlinearity = nonlinearity
        # Boolean flag to determine if CRF is to be used
        self.viterbi = viterbi
        # Vocab size
        self.vocab_size = vocab_size
        # Shape domain size
        self.shape_domain_size = shape_domain_size
        # Shape Size
        self.shape_size = shape_size
        # Whether we need to use character embeddings as input
        self.use_characters = self.char_size != 0
        # Whether we need to use shape embeddings as input
        self.use_shape = self.shape_size != 0
        # Declaring the word embedding shape
        self.word_embeddings_shape = (vocab_size - 1, embedding_size)
        ################################################################################
        # Initializing the TensorFlow variables
        ################################################################################

        # PLACEHOLDERS
        # word embedding input : 2-D int matrix
        self.input_x1 = tf.placeholder(tf.int64, [None, None], name="input_x1")
        # shape embedding input
        self.input_x2 = tf.placeholder(tf.int64, [None, None], name="input_x2")
        # labels : 2-D int matrix
        self.input_y = tf.placeholder(tf.int64, [None, None], name="input_y")
        # padding mask
        self.input_mask = tf.placeholder(tf.float32, [None, None], name="input_mask")
        # Batch size
        self.batch_size = tf.placeholder(tf.int32, None, name="batch_size")
        # Maximum sequence length
        self.max_seq_len = tf.placeholder(tf.int32, None, name="max_seq_len")
        # sequence lengths padded with zeros
        self.sequence_lengths = tf.placeholder(tf.int32, [None, None], name="sequence_lengths")
        # dropout and l2 penalties
        self.middle_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="middle_dropout_keep_prob")
        self.hidden_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="hidden_dropout_keep_prob")
        self.input_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="input_dropout_keep_prob")
        self.l2_penalty = tf.placeholder_with_default(0.0, [], name="l2_penalty")
        self.projection = tf.placeholder_with_default(False, [], name="projection")
        self.drop_penalty = tf.placeholder_with_default(0.0, [], name="drop_penalty")

        # CONSTANTS
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)
        # set the pad token to a constant 0 vector
        self.word_zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, embedding_size])
        self.shape_zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, shape_size])
        self.char_zero_pad = tf.constant(0.0, dtype=tf.float32, shape=[1, char_size])
        # Declaring transition probabilities to use when it is CRF
        if self.viterbi:
            self.transition_params = tf.get_variable("transitions", [num_classes, num_classes])
        # Initializing Word-Embedding layer
        self.w_e = tf_utils.initialize_embeddings(self.word_embeddings_shape, name="w_e", pretrained=embeddings)

        # Since 0 is padding : calculating non-zero values for masking
        nonzero_elements = tf.not_equal(self.sequence_lengths, tf.zeros_like(self.sequence_lengths))
        count_nonzero_per_row = tf.reduce_sum(tf.to_int32(nonzero_elements), axis=1)
        self.flat_sequence_lengths = tf.add(tf.reduce_sum(self.sequence_lengths, 1),
                                            tf.scalar_mul(2, count_nonzero_per_row))

        # Calling the forward layer
        self.unflat_scores, self.hidden_layer = self.forward(self.input_x1, self.input_x2, self.max_seq_len,
                                                             self.hidden_dropout_keep_prob,
                                                             self.input_dropout_keep_prob,
                                                             self.middle_dropout_keep_prob, reuse=False)
        # Calling loss function
        self.loss = self.get_loss()

        # Getting predictions
        self.predictions = self.get_predictions()

    def get_loss(self):
        """
        Calculate mean cross-entropy loss
        :return:
        """
        with tf.name_scope("loss"):
            labels = tf.cast(self.input_y, 'int32')
            if self.viterbi:
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.unflat_scores, labels,
                                                                                      self.flat_sequence_lengths,
                                                                                      transition_params=self.transition_params)
                # self.transition_params = transition_params
                loss = tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.unflat_scores, labels=labels)
                masked_losses = tf.multiply(losses, self.input_mask)
                loss = tf.div(tf.reduce_sum(masked_losses), tf.reduce_sum(self.input_mask))
            loss += self.l2_penalty * self.l2_loss
            unflat_no_dropout_scores, _ = self.forward(self.input_x1, self.input_x2, self.max_seq_len,
                                                       hidden_dropout_keep_prob=1.0, input_dropout_keep_prob=1.0,
                                                       middle_dropout_keep_prob=1.0)
            drop_loss = tf.nn.l2_loss(tf.subtract(self.unflat_scores, unflat_no_dropout_scores))
            loss += self.drop_penalty * drop_loss
        return loss

    def get_predictions(self):
        """
        Get prediction scores
        :return:
        """
        with tf.name_scope("predictions"):
            if self.viterbi:
                predictions_scores = self.unflat_scores
            else:
                predictions_scores = tf.argmax(self.unflat_scores, 2)
        return predictions_scores

    def forward(self, input_x1, input_x2, max_seq_len, hidden_dropout_keep_prob,
                input_dropout_keep_prob, middle_dropout_keep_prob, reuse=True):
        """
        Passing the inputs through the forward layer
        :param input_x1:
        :param max_seq_len:
        :param hidden_dropout_keep_prob:
        :param input_dropout_keep_prob:
        :param middle_dropout_keep_prob:
        :param reuse:
        :return:
        """
        word_embeddings = tf.nn.embedding_lookup(self.w_e, input_x1)
        with tf.variable_scope("forward", reuse=reuse):
            input_list = [word_embeddings]
            input_size = self.embedding_size
            if self.use_characters:
                input_list.append(self.char_embeddings)
                input_size += self.char_size
            if self.use_shape:
                shape_embeddings_shape = (self.shape_domain_size - 1, self.shape_size)
                w_s = tf_utils.initialize_embeddings(shape_embeddings_shape, name="w_s")
                shape_embeddings = tf.nn.embedding_lookup(w_s, input_x2)
                input_list.append(shape_embeddings)
                input_size += self.shape_size

            input_feats = tf.concat(axis=2, values=input_list)
            input_feats_expanded_drop = tf.nn.dropout(input_feats, input_dropout_keep_prob)
            total_output_width = 2 * self.hidden_dim
            with tf.variable_scope("bilstm", reuse=reuse):
                fwd_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
                bwd_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
                lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell, cell_bw=bwd_cell, dtype=tf.float32,
                                                                  inputs=input_feats_expanded_drop,
                                                                  parallel_iterations=50,
                                                                  sequence_length=self.flat_sequence_lengths)
                hidden_outputs = tf.concat(axis=2, values=lstm_outputs)

            h_concat_flat = tf.reshape(hidden_outputs, [-1, total_output_width])

            # Add dropout
            with tf.name_scope("middle_dropout"):
                h_drop = tf.nn.dropout(h_concat_flat, middle_dropout_keep_prob)

            # second projection
            with tf.name_scope("tanh_proj"):
                w_tanh = tf_utils.initialize_weights([total_output_width, self.hidden_dim], "w_tanh",
                                                     init_type="xavier")
                b_tanh = tf.get_variable(initializer=tf.constant(0.01, shape=[self.hidden_dim]), name="b_tanh")
                self.l2_loss += tf.nn.l2_loss(w_tanh)
                self.l2_loss += tf.nn.l2_loss(b_tanh)
                h2_concat_flat = tf.nn.xw_plus_b(h_drop, w_tanh, b_tanh, name="h2_tanh")
                h2_tanh = tf_utils.apply_nonlinearity(h2_concat_flat, self.nonlinearity)

            # Add dropout
            with tf.name_scope("hidden_dropout"):
                h2_drop = tf.nn.dropout(h2_tanh, hidden_dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                w_o = tf_utils.initialize_weights([self.hidden_dim, self.num_classes], "w_o", init_type="xavier")
                b_o = tf.get_variable(initializer=tf.constant(0.01, shape=[self.num_classes]), name="b_o")
                self.l2_loss += tf.nn.l2_loss(w_o)
                self.l2_loss += tf.nn.l2_loss(b_o)
                scores = tf.nn.xw_plus_b(h2_drop, w_o, b_o, name="scores")
                unflat_scores = tf.reshape(scores, tf.stack([self.batch_size, max_seq_len, self.num_classes]))
        return unflat_scores, hidden_outputs
