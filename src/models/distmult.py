import numpy as np
import random
import sys

from src.models.text_encoders import *


class _KBModel(object):
    """
    Class which models the Knowledge graph (consisting of triplets)
    """

    def __init__(self, lr, embed_dim, token_dim, lstm_dim,
                 entity_vocab_size, kb_vocab_size, token_vocab_size,
                 loss_type, margin, l2_weight, neg_samples,
                 use_tanh=True, max_pool=True, bidirectional=True, peephole=False, verbose=False, freeze=False):
        # Defining basic object variables
        self._lr = lr  # Learning rate
        self._embed_dim = embed_dim  # Embedding dimension
        self._token_dim = token_dim  # Token Dimension
        self._lstm_dim = lstm_dim  # Dimension of the LSTM
        self._kb_size = kb_vocab_size  # Size of the knowledge base vocabulary : basically number of relations
        self._token_size = token_vocab_size  # Size of the tokens in knowledge base
        self._peephole = peephole
        self._epsilon = tf.constant(0.00001, name='epsilon')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.entity_size = entity_vocab_size
        self.non_linear = tf.nn.tanh if use_tanh else tf.identity  # Non-linearity
        self.max_pool = max_pool  # Using max pool
        self.bidirectional = bidirectional  # whether to use bidirectional lstm or not
        self.margin = margin  # Margin for Margin loss
        self.verbose = verbose
        self.freeze = freeze

        # set up placeholders
        self.text_update = tf.placeholder_with_default(True, [], name='text_update')
        self.loss_weight = tf.placeholder_with_default(1.0, [], name='loss_weight')
        self.variance_weight = tf.placeholder_with_default(0.0, [], name='var_weight')
        self.kb_batch = tf.placeholder_with_default([0], [None], name='kb_batch')
        self.text_batch = tf.placeholder_with_default([[0]], [None, None], name='text_batch')
        self.seq_len_batch = tf.placeholder_with_default([0], [None], name='seq_len_batch')
        self.word_dropout_keep = tf.placeholder_with_default(1.0, [], name='word_keep_prob')
        self.lstm_dropout_keep = tf.placeholder_with_default(1.0, [], name='lstm_keep_prob')
        self.final_dropout_keep = tf.placeholder_with_default(1.0, [], name='final_keep_prob')
        self.neg_samples = tf.placeholder_with_default(neg_samples, [], name='neg_samples')

        # initialize embedding tables
        self.token_embeddings = tf.get_variable(name='token_embeddings',
                                                shape=[self._token_size, self._token_dim],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                trainable=(not freeze)
                                                )
        self.kb_embeddings = tf.get_variable(name='kb_embeddings',
                                             shape=[self._kb_size, self._embed_dim],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             trainable=(not freeze)
                                             )

        self.text_encoder = LSTMTextEncoder(text_placeholder=self.text_batch,
                                            seq_len_placeholder=self.seq_len_batch,
                                            lstm_dim=self._lstm_dim,
                                            embed_dim=self._embed_dim, token_dim=self._token_dim,
                                            word_dropout_keep=self.word_dropout_keep,
                                            lstm_dropout_keep=self.lstm_dropout_keep,
                                            final_dropout_keep=self.final_dropout_keep, entity_index=100,
                                            use_birectional=True)

        self.renorm = tf.no_op()
        # TO get the cosine similarity of the knowledge embedding with
        self.kb_nearest_neighbor = tf_utils.embedding_nearest_neighbors(self.kb_embeddings, self.kb_batch)

    def embed_text(self, token_embeddings, scope_name='text', reuse=False):
        return self.text_encoder.embed_text(token_embeddings=token_embeddings, scope_name=scope_name, reuse=reuse)

    def embed_kb(self, kb_embeddings, scope_name='kb', reuse=False):
        '''
            Lookup kb relation embeddings directly in embedding table
        :return: a batch of kb relation embeddings
        '''
        with tf.variable_scope(scope_name, reuse=reuse):
            selected_col_embeddings = tf.nn.embedding_lookup(kb_embeddings, self.kb_batch)
            return selected_col_embeddings

    def semi_hard_negatives(self, sess, feed_dict, neg_batch_key, neg_pred_key,
                            neg_entities, neg_samples, neg_samples_plus):
        feed_dict[neg_batch_key] = neg_entities
        feed_dict[self.neg_samples] = neg_samples_plus

        pos_scores, neg_scores = sess.run([self.pos_predictions, neg_pred_key], feed_dict=feed_dict)
        pos_scores_repeated = np.repeat(pos_scores, neg_samples_plus, axis=1)

        c1 = np.less(neg_scores, pos_scores_repeated)
        c2 = np.less(pos_scores_repeated, neg_scores + self.margin)
        c3 = np.squeeze(np.logical_and(c1, c2))
        _neg_subset = [neg_entities[i, c3[i]] for i in range(c3.shape[0])]
        neg_subset = [s[:neg_samples] if len(s) >= neg_samples
                      else np.concatenate([s, np.random.randint(low=0, high=self.entity_size,
                                                                size=(neg_samples - len(s)))], axis=0)
                      for s in _neg_subset]
        feed_dict[neg_batch_key] = neg_subset
        feed_dict[self.neg_samples] = neg_samples
        return feed_dict

    # Losses
    def hinge_loss(self, pos_predictions, neg_predictions, margin):
        '''
               compute hinge loss between positive and negative triple,
        '''
        err = tf.nn.relu(neg_predictions - pos_predictions + tf.constant(margin))
        # avg over the err for each negative sample
        err = tf.reduce_mean(err, axis=[1])
        return err

    def sampled_softmax_loss(self, pos_predictions, neg_predictions, labels):
        '''
            Compute softmax loss over sampled negatives
        '''
        logits = tf.squeeze(tf.concat(axis=1, values=[pos_predictions, neg_predictions]), axis=[2])
        err = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return err


class _EntityModel(_KBModel):
    """
    Class which tries to create a model for entities
    """

    def __init__(self, lr, embed_dim, token_dim, lstm_dim,
                 entity_vocab_size, kb_vocab_size, token_vocab_size,
                 loss_type, margin, l2_weight, neg_samples, norm_entities,
                 use_tanh=True, max_pool=True, bidirectional=True, peephole=False, verbose=False, freeze=False):

        super(_EntityModel, self).__init__(lr=lr, embed_dim=embed_dim, token_dim=token_dim, lstm_dim=lstm_dim,
                                           entity_vocab_size=entity_vocab_size,
                                           kb_vocab_size=kb_vocab_size, token_vocab_size=token_vocab_size,
                                           loss_type=loss_type, margin=margin, l2_weight=l2_weight,
                                           neg_samples=neg_samples,
                                           use_tanh=use_tanh, max_pool=max_pool, bidirectional=bidirectional,
                                           peephole=peephole, verbose=verbose, freeze=freeze)
        self.model_type = 'entity_model'
        self.entity_size = entity_vocab_size
        self.e1_batch = tf.placeholder(tf.int32, [None], name='e1_batch')
        self.e2_batch = tf.placeholder(tf.int32, [None], name='e2_batch')
        self.neg_e1_batch = tf.placeholder(tf.int32, [None, None], name='neg_e1_batch')
        self.neg_e2_batch = tf.placeholder(tf.int32, [None, None], name='neg_e2_batch')
        self.entity_embeddings = tf.get_variable(name='row_embeddings',
                                                 shape=[self.entity_size, self._embed_dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())

        # encode row and col
        pos_e1_embeddings, pos_e2_embeddings, rel_embeddings = \
            self.get_embeddings(entity_embeddings=self.entity_embeddings, kb_embeddings=self.kb_embeddings)

        # calculate err
        self.pos_predictions, self.neg_predictions_1, self.neg_predictions_2 = self.generate_predictions(
            pos_e1_embeddings, pos_e2_embeddings, rel_embeddings, self.neg_samples)

        if loss_type == 'hinge':
            self.reconstruction_loss = self.hinge_loss(self.pos_predictions, self.neg_predictions_1, margin) \
                                       + self.hinge_loss(self.pos_predictions, self.neg_predictions_2, margin)
        elif loss_type == 'softmax':
            zero_batch = tf.zeros_like(self.e1_batch)
            self.reconstruction_loss = self.sampled_softmax_loss(self.pos_predictions, self.neg_predictions_1,
                                                                 zero_batch) \
                                       + self.sampled_softmax_loss(self.pos_predictions, self.neg_predictions_2,
                                                                   zero_batch)
        else:
            print('loss type %s is not a valid loss type' % loss_type)
            sys.exit(1)

        # weight kb and text loss differently
        l2_loss = tf.constant(l2_weight) * tf.nn.l2_loss(rel_embeddings)
        self.loss = self.calculate_loss(self.reconstruction_loss, l2_loss)

        # keep entities at norm 1
        if norm_entities:
            self.renorm = tf.assign(self.entity_embeddings,
                                    tf.clip_by_norm(self.entity_embeddings, clip_norm=1, axes=1), name='renorm')
        self.entity_nearest_neighbor = tf_utils.embedding_nearest_neighbors(self.entity_embeddings, self.e1_batch)

    def get_embeddings(self, entity_embeddings, kb_embeddings):
        pos_e1_embeddings = tf.nn.embedding_lookup(entity_embeddings, self.e1_batch)
        pos_e2_embeddings = tf.nn.embedding_lookup(entity_embeddings, self.e2_batch)
        rel_embeddings = tf.no_op()
        if self.text_encoder.encoder_type == 'lstm':
            kb_embedding = self.embed_kb(kb_embeddings)
            rel_embeddings = kb_embedding

        return pos_e1_embeddings, pos_e2_embeddings, rel_embeddings

    # data helper methods
    def generate_negatives(self, entity_batch, max_entity, neg_samples):
        neg_batch = []
        for i in range(neg_samples):
            neg_sample = []
            for e in entity_batch:
                neg_e = random.randint(0, max_entity - 1)
                neg_sample.append(neg_e)
            neg_batch.append(neg_sample)
        return np.transpose(np.array(neg_batch), (1, 0))

    def add_entities_to_feed_dict(self, sess, feed_dict, batch, neg_samples, semi_hard=False):
        e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len = batch

        feed_dict[self.e1_batch] = e1
        feed_dict[self.e2_batch] = e2

        if semi_hard:
            neg_samples_plus = neg_samples * 5
            neg_ents = self.generate_negatives(e1, self.entity_size, neg_samples_plus)
            feed_dict = self.semi_hard_negatives(sess, feed_dict, self.neg_e1_batch, self.neg_predictions_1,
                                                 neg_ents, neg_samples, neg_samples_plus)
            feed_dict = self.semi_hard_negatives(sess, feed_dict, self.neg_e2_batch, self.neg_predictions_2,
                                                 neg_ents, neg_samples, neg_samples_plus)
        else:
            neg_e1 = self.generate_negatives(e1, self.entity_size, neg_samples)
            neg_e2 = self.generate_negatives(e2, self.entity_size, neg_samples)
            feed_dict[self.neg_e1_batch] = neg_e1
            feed_dict[self.neg_e2_batch] = neg_e2

        return feed_dict

    # Abstract methods
    def generate_predictions(self, pos_e1_embeddings, pos_e2_embeddings, selected_col_embeddings, neg_samples):
        print('Error, this is an abstract class and does not implement \'generate_predictions()\'')
        sys.exit(1)
        return None, None, None

    def calculate_loss(self, reconstruction_loss, l2_loss):
        return self.loss_weight * (tf.reduce_sum(reconstruction_loss) + l2_loss)


class DistMult(_EntityModel):
    def generate_predictions(self, pos_e1_embeddings, pos_e2_embeddings, selected_col_embeddings, neg_samples):
        rel = tf.expand_dims(self.non_linear(selected_col_embeddings), 2)

        neg_e1_flat = tf.nn.embedding_lookup(self.entity_embeddings, tf.reshape(self.neg_e1_batch, [-1]))
        neg_e2_flat = tf.nn.embedding_lookup(self.entity_embeddings, tf.reshape(self.neg_e2_batch, [-1]))

        neg_e1_embeddings = self.non_linear(tf.reshape(neg_e1_flat, [-1, neg_samples, self._embed_dim]))
        neg_e2_embeddings = self.non_linear(tf.reshape(neg_e2_flat, [-1, neg_samples, self._embed_dim]))
        pos_e1_expanded = self.non_linear(tf.expand_dims(pos_e1_embeddings, 1))
        pos_e2_expanded = self.non_linear(tf.expand_dims(pos_e2_embeddings, 1))

        pos_ep = tf.multiply(pos_e1_expanded, pos_e2_expanded)
        neg_ep1 = tf.multiply(neg_e1_embeddings, pos_e2_expanded)
        neg_ep2 = tf.multiply(pos_e1_expanded, neg_e2_embeddings)

        pos_predictions = tf.matmul(pos_ep, rel)
        neg_predictions_1 = tf.matmul(neg_ep1, rel)
        neg_predictions_2 = tf.matmul(neg_ep2, rel)

        return pos_predictions, neg_predictions_1, neg_predictions_2
