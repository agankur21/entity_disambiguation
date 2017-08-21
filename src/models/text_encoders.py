from __future__ import print_function
from __future__ import division
import tensorflow as tf
import src.utils.tf_utils as tf_utils


class TextEncoder(object):
    def __init__(self, text_placeholder, seq_len_placeholder, lstm_dim, embed_dim, token_dim, word_dropout_keep, lstm_dropout_keep,
                 final_dropout_keep,
                 entity_index=100):
        self.text_placeholder = text_placeholder
        self.seq_len_placeholder = seq_len_placeholder
        self.lstm_dim = lstm_dim
        self.embed_dim = embed_dim
        self.token_dim = token_dim
        self.word_dropout_keep = word_dropout_keep
        self.lstm_dropout_keep = lstm_dropout_keep
        self.final_dropout_keep = final_dropout_keep
        self.entity_index = entity_index

    def get_token_embeddings(self, token_embeddings):
        selected_words = tf.nn.embedding_lookup(token_embeddings, self.text_placeholder)
        token_embeds = selected_words
        dropped_embeddings = tf.nn.dropout(token_embeds, self.word_dropout_keep)
        return dropped_embeddings

    def concat_and_project(self, inputs, output_projection, bias, dropout_keep_prob):
        output_state = tf.concat(axis=1, values=inputs)
        dropped_output_state = tf.nn.dropout(output_state, dropout_keep_prob)
        return tf.nn.xw_plus_b(dropped_output_state, output_projection, bias)


class LSTMTextEncoder(TextEncoder):
    def __init__(self, text_placeholder, seq_len_placeholder, lstm_dim, embed_dim,
                 token_dim, word_dropout_keep, lstm_dropout_keep, final_dropout_keep, entity_index=100,
                 use_birectional=True):
        super(LSTMTextEncoder, self).__init__(text_placeholder, seq_len_placeholder, lstm_dim,
                                              embed_dim, token_dim,
                                              word_dropout_keep, lstm_dropout_keep, final_dropout_keep, entity_index)
        self.encoder_type = 'lstm'
        self.bidirectional = use_birectional
        self.peephole = True

    def embed_text(self, token_embeddings, scope_name='lstm_text', reuse=False):
        '''
            Use LSTM to encode sentence
        '''
        selected_col_embeddings = self.get_token_embeddings(token_embeddings)
        return self.forward(selected_col_embeddings, self.seq_len_placeholder, scope_name=scope_name, reuse=reuse)

    def forward(self, selected_col_embeddings, seq_len, scope_name='text', reuse=False):
        '''
        concat last output of forward and backward lstms and project
        '''
        with tf.variable_scope(scope_name, reuse=reuse):
            with tf.variable_scope('fw', reuse=reuse):
                fw_cell = tf.contrib.rnn.LSTMCell(self.lstm_dim, use_peepholes=self.peephole,
                                                  num_proj=self.embed_dim, state_is_tuple=True)
            if not self.bidirectional:
                outputs, state = tf.nn.dynamic_rnn(cell=fw_cell,
                                                   inputs=selected_col_embeddings,
                                                   sequence_length=seq_len,
                                                   dtype=tf.float32,
                                                   parallel_iterations=5000,
                                                   )
                return tf_utils.last_relevant(outputs, seq_len)

            else:
                with tf.variable_scope('bw', reuse=reuse):
                    bw_cell = tf.contrib.rnn.LSTMCell(self.lstm_dim, use_peepholes=self.peephole,
                                                      num_proj=self.embed_dim, state_is_tuple=True)
                lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                  dtype=tf.float32,
                                                                  inputs=selected_col_embeddings,
                                                                  parallel_iterations=5000,
                                                                  sequence_length=seq_len)
                fw_output = tf_utils.last_relevant(lstm_outputs[0], seq_len)
                bw_output = lstm_outputs[1][:, 0, :]
                output_projection = tf.get_variable(name='lstm_output_projection',
                                                    shape=[self.embed_dim * 2, self.embed_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())
                output_bias = tf.get_variable(name='lstm_projection_bias',
                                              initializer=tf.constant(0.0001, shape=[self.embed_dim]))
                outputs = self.concat_and_project([fw_output, bw_output], output_projection,
                                                  output_bias, self.lstm_dropout_keep)
                return outputs
