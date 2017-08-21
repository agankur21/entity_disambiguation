import tensorflow as tf
import random
import sys
from src.models.entity_models import _EntityModel


class RNNEntity(_EntityModel):
    def embed_text(self, token_embeddings, position_embeddings, ep_embeddings, scope_name='text', reuse=False):
        text_encoder = self.text_encoder
        selected_col_embeddings = text_encoder.get_token_embeddings(token_embeddings, position_embeddings)
        batch_size = tf.shape(selected_col_embeddings)[0]
        max_seq_len = tf.shape(selected_col_embeddings)[1]
        encoded_tokens = text_encoder.forward(selected_col_embeddings, batch_size, max_seq_len, reuse=reuse)
        e1_embed = text_encoder.avg_entity_tokens(encoded_tokens, self.e1_dist_batch, batch_size, max_seq_len)
        e2_embed = text_encoder.avg_entity_tokens(encoded_tokens, self.e2_dist_batch, batch_size, max_seq_len)
        ep_embed = self.embed_entities([e1_embed, e2_embed], reuse=reuse)
        return ep_embed

    def embed_entities(self, entity_embeddings, reuse=False):
        with tf.variable_scope('entity_rnn', reuse=reuse):
            expanded_entities = [tf.expand_dims(e, 1) for e in entity_embeddings]
            entity_tensor = tf.concat(axis=1, values=expanded_entities)
            # entity_rnn = tf.nn.rnn_cell.BasicRNNCell(self._embed_dim)
            entity_rnn = tf.nn.rnn_cell.LSTMCell(self._lstm_dim, use_peepholes=self._peephole,
                                                 num_proj=self._embed_dim, state_is_tuple=True)
            outputs, state = tf.nn.dynamic_rnn(cell=entity_rnn,
                                               inputs=entity_tensor,
                                               dtype=tf.float32,
                                               parallel_iterations=5000,
                                               )
            row_embedding = outputs[:, -1, :]
            return row_embedding

    def generate_predictions(self, pos_e1_embeddings, pos_e2_embeddings, selected_col_embeddings, neg_samples):
        # TODO: only support 1 negative sample
        rel = tf.expand_dims(self.non_linear(selected_col_embeddings), 2)

        neg_e1_flat = tf.nn.embedding_lookup(self.entity_embeddings, tf.reshape(self.neg_e1_batch, [-1]))
        neg_e2_flat = tf.nn.embedding_lookup(self.entity_embeddings, tf.reshape(self.neg_e2_batch, [-1]))

        neg_e1_embeddings = self.non_linear(tf.reshape(neg_e1_flat, [-1, self._embed_dim]))
        neg_e2_embeddings = self.non_linear(tf.reshape(neg_e2_flat, [-1, self._embed_dim]))

        pos_e1_non_linear = self.non_linear(pos_e1_embeddings)
        pos_e2_non_linear = self.non_linear(pos_e2_embeddings)

        pos_ep = self.embed_entities([pos_e1_non_linear, pos_e2_non_linear], reuse=True)
        pos_ep = tf.expand_dims(pos_ep, 1)

        neg_ep1 = self.embed_entities([neg_e1_embeddings, pos_e2_non_linear], reuse=True)
        neg_ep1 = tf.expand_dims(neg_ep1, 1)
        neg_ep2 = self.embed_entities([pos_e1_non_linear, neg_e2_embeddings], reuse=True)
        neg_ep2 = tf.expand_dims(neg_ep2, 1)

        pos_predictions = tf.matmul(pos_ep, rel)
        neg_predictions_1 = tf.matmul(neg_ep1, rel)
        neg_predictions_2 = tf.matmul(neg_ep2, rel)

        return pos_predictions, neg_predictions_1, neg_predictions_2

    def calculate_loss(self, reconstruction_loss, l2_loss, e1_embeddings, e2_embeddings, rel_embeddings, variance_type):
        if variance_type == 'None':
            return self.loss_weight * (tf.reduce_sum(reconstruction_loss) + l2_loss)
        else:
            return (tf.cond(self.text_update,
                            lambda: self.variance_loss(reconstruction_loss, l2_loss, rel_embeddings, variance_type),
                            lambda: self.kb_variance_loss(reconstruction_loss, l2_loss, e1_embeddings, e2_embeddings,
                                                          variance_type, reuse=True)))

    def variance_loss(self, reconstruction_loss, l2_loss, rel_embeddings, variance_type):
        noise_prediction = self.variance(rel_embeddings, variance_type)
        weighted_err = tf.multiply(tf.constant(1.0) - noise_prediction, reconstruction_loss)
        var_penalty = tf.multiply(self.variance_weight, noise_prediction)
        if self.verbose:
            var_penalty = tf.Print(var_penalty, [tf.reduce_sum(weighted_err), tf.reduce_sum(reconstruction_loss),
                                                 tf.reduce_sum(var_penalty)],  message='penalties')
            var_penalty = tf.Print(var_penalty, [tf.reduce_mean(noise_prediction), tf.reduce_max(noise_prediction),
                                                 tf.reduce_min(noise_prediction)], message='noise')
            var_penalty = tf.Print(var_penalty, [self.variance_weight, tf.reduce_mean(var_penalty),
                                                 tf.reduce_max(var_penalty), tf.reduce_min(var_penalty)], message='variance')
            var_penalty = tf.Print(var_penalty, [tf.reduce_mean(reconstruction_loss), tf.reduce_max(reconstruction_loss),
                                                 tf.reduce_min(reconstruction_loss)], message='err')

        return self.loss_weight * (tf.reduce_sum(weighted_err)
                                   + l2_loss
                                   + tf.reduce_sum(var_penalty))

    def kb_variance_loss(self, reconstruction_loss, l2_loss, e1_embeddings, e2_embeddings, variance_type, reuse):
        ep = self.embed_entities([self.non_linear(e1_embeddings), self.non_linear(e2_embeddings)], reuse=True)
        noise_prediction = self.variance(ep, variance_type, reuse=reuse)
        l2_loss_weight = 10.0
        kb_target = 1.0
        kb_prediction_loss = l2_loss_weight * tf.nn.l2_loss(1.0 - (kb_target - noise_prediction))
        if self.verbose:
            kb_prediction_loss = tf.Print(kb_prediction_loss, [tf.reduce_sum(reconstruction_loss), kb_prediction_loss],
                                          message='kb prediction')
        return self.loss_weight * (tf.reduce_sum(reconstruction_loss)
                                   + l2_loss
                                   + kb_prediction_loss)

    def variance(self, embedding, variance_type, scope_name='variance', reuse=False):
        variance_non_linear = tf.nn.relu
        with tf.variable_scope(scope_name, reuse=reuse):
            var_matrix_rel_0 = tf.get_variable(name='var_rel_matrix_0', shape=[self._embed_dim, self._embed_dim],
                                               initializer=tf.contrib.layers.xavier_initializer())
            var_bias_rel_0 = tf.get_variable(name='var_bias_rel_0',
                                             initializer=tf.constant(0.0001, shape=[self._embed_dim]))
            var_matrix_rel = tf.get_variable(name='var_rel_matrix', shape=[self._embed_dim, 1],
                                             initializer=tf.contrib.layers.xavier_initializer())
            var_bias_rel = tf.get_variable(name='var_bias_rel', initializer=tf.constant(0.0001, shape=[1]))
        _noise_prediction_0 = tf.nn.xw_plus_b(variance_non_linear(embedding),
                                              var_matrix_rel_0, var_bias_rel_0)
        _noise_prediction_1 = tf.nn.xw_plus_b(variance_non_linear(_noise_prediction_0),
                                              var_matrix_rel, var_bias_rel)
        noise_prediction = tf.nn.sigmoid(_noise_prediction_1)
        return noise_prediction


class PathRNNEntity(RNNEntity):
    def __init__(self, lr, embed_dim, token_dim, lstm_dim, position_dim,
                 ep_vocab_size, entity_vocab_size, kb_vocab_size, token_vocab_size, position_vocab_size,
                 loss_type, margin, l2_weight, neg_samples, norm_entities, text_encoder, variance_type,
                 use_tanh=True, max_pool=True, bidirectional=True, peephole=False, verbose=False, freeze=False):

        super(PathRNNEntity, self).__init__(lr, embed_dim, token_dim, lstm_dim, position_dim,
                                            ep_vocab_size, entity_vocab_size, kb_vocab_size, token_vocab_size,
                                            position_vocab_size,
                                            loss_type, margin, l2_weight, neg_samples, norm_entities, text_encoder,
                                            variance_type,
                                            use_tanh, max_pool, bidirectional, peephole, verbose, freeze)
        self.path_batch = tf.placeholder_with_default([[[0]]], [None, None, None], name='path_batch')
        self.neg_path_batch = tf.placeholder_with_default([[[0]]], [None, None, None], name='neg_path_batch')
        self.path_seq_len_batch = tf.placeholder_with_default([[0]], [None, None], name='path_seq_len_batch')
        self.num_paths = tf.placeholder_with_default(1, [], name='num_paths')
        self.max_path_len = tf.placeholder_with_default(1, [], name='max_path_len')
        self.path_loss = self.path_update()

    def path_update(self):
        path_lens = tf.reshape(self.path_seq_len_batch, [-1])
        kb_embeddings = tf.nn.embedding_lookup(self.kb_embeddings, self.kb_batch)
        pos_predictions = self.path_predictions(kb_embeddings, self.path_batch, path_lens)
        neg_predictions = self.path_predictions(kb_embeddings, self.neg_path_batch, path_lens)
        loss = self.hinge_loss(pos_predictions, neg_predictions, self.margin)
        return tf.reduce_sum(loss)

    def path_predictions(self, kb_embeddings, path_batch, path_lens):
        flat_path_batch = tf.reshape(path_batch, [-1, self.max_path_len])
        path_entity_embeddings = tf.nn.embedding_lookup(self.entity_embeddings, flat_path_batch)
        encoded_paths = self.encode_path(path_entity_embeddings, path_lens, reuse=True)
        expanded_encoded_path = tf.expand_dims(encoded_paths, 2)

        predictions = tf.matmul(tf.expand_dims(kb_embeddings, 1), expanded_encoded_path)
        prediction_reshaped = tf.reshape(predictions, [-1, self.num_paths])
        predictions_aggregated = tf.reduce_logsumexp(prediction_reshaped, axis=[1])
        predictions_aggregated_expanded = tf.expand_dims(predictions_aggregated, 1)
        return predictions_aggregated_expanded

    def embed_entities(self, entity_embeddings, reuse=False):
        expanded_entities = [tf.expand_dims(e, 1) for e in entity_embeddings]
        entity_tensor = tf.concat(axis=1, values=expanded_entities)
        return self.encode_path(entity_tensor, None, reuse)

    def encode_path(self, entity_tensor, path_lens, reuse=False):
        with tf.variable_scope('entity_rnn', reuse=reuse):
            entity_rnn = tf.nn.rnn_cell.BasicRNNCell(self._embed_dim)
            outputs, state = tf.nn.dynamic_rnn(cell=entity_rnn,
                                               inputs=entity_tensor,
                                               dtype=tf.float32,
                                               sequence_length=path_lens,
                                               parallel_iterations=5000,
                                               )
        row_embedding = outputs[:, -1, :]
        return row_embedding


class RNNEntityWithTypes(RNNEntity):
    def __init__(self, lr, embed_dim, token_dim, lstm_dim, position_dim,
                 ep_vocab_size, entity_vocab_size, kb_vocab_size, token_vocab_size, position_vocab_size,
                 loss_type, margin, l2_weight, neg_samples, norm_entities, text_encoder, variance_type,
                 use_tanh=True, max_pool=True, bidirectional=True, peephole=False, verbose=False, freeze=False):
        # TODO: dont hardcode type vocab size
        self.margin = margin
        self.type_id_map = None
        self.type_vocab_size = 817
        self.entity_type_embeddings = tf.get_variable(name='entity_type_embeddings',
                                                      shape=[self.type_vocab_size + 1, embed_dim],
                                                      initializer=tf.contrib.layers.xavier_initializer())
        self.e1_type_batch = tf.placeholder(tf.int32, [None], name='e1_type_batch')
        self.e2_type_batch = tf.placeholder(tf.int32, [None], name='e2_type_batch')
        self.e1_neg_type_batch = tf.placeholder(tf.int32, [None], name='e1_neg_type_batch')
        self.e2_neg_type_batch = tf.placeholder(tf.int32, [None], name='e2_neg_type_batch')

        super(RNNEntityWithTypes, self).__init__(lr, embed_dim, token_dim, lstm_dim, position_dim,
                                                 ep_vocab_size, entity_vocab_size, kb_vocab_size, token_vocab_size,
                                                 position_vocab_size,
                                                 loss_type, margin, l2_weight, neg_samples, norm_entities, text_encoder,
                                                 variance_type,
                                                 use_tanh, max_pool, bidirectional, peephole, verbose, freeze)

    def entity_type_loss(self, e1_embeddings, e2_embeddings):
        pos_e1_non_linear = tf.expand_dims(self.non_linear(e1_embeddings), 2)
        pos_e2_non_linear = tf.expand_dims(self.non_linear(e2_embeddings), 2)
        # lookup type embeddings
        e1_type_embed = self.non_linear(
                tf.expand_dims(tf.nn.embedding_lookup(self.entity_type_embeddings, self.e1_type_batch), 1))
        e1_neg_type = self.non_linear(
                tf.expand_dims(tf.nn.embedding_lookup(self.entity_type_embeddings, self.e1_neg_type_batch), 1))
        e2_type_embed = self.non_linear(
                tf.expand_dims(tf.nn.embedding_lookup(self.entity_type_embeddings, self.e2_type_batch), 1))
        e2_neg_type = self.non_linear(
                tf.expand_dims(tf.nn.embedding_lookup(self.entity_type_embeddings, self.e2_neg_type_batch), 1))

        print(e1_type_embed.get_shape())
        print(pos_e1_non_linear.get_shape())
        pos_e1 = tf.matmul(e1_type_embed, pos_e1_non_linear)
        pos_e2 = tf.matmul(e2_type_embed, pos_e2_non_linear)
        neg_e1 = tf.matmul(e1_neg_type, pos_e1_non_linear)
        neg_e2 = tf.matmul(e2_neg_type, pos_e2_non_linear)
        entity_loss = self.hinge_loss(pos_e1, neg_e1, self.margin) + self.hinge_loss(pos_e2, neg_e2, self.margin)
        return entity_loss

    def calculate_loss(self, reconstruction_loss, l2_loss, e1_embeddings,
                       e2_embeddings, rel_embeddings, variance_type):
        type_loss_weight = .01
        entity_type_loss = tf.reduce_sum(self.entity_type_loss(e1_embeddings, e2_embeddings))*type_loss_weight
        loss = self.loss_weight * (tf.reduce_sum(reconstruction_loss) + l2_loss + entity_type_loss)
        loss = tf.Print(loss, [tf.reduce_sum(reconstruction_loss), l2_loss, entity_type_loss],
                        message='losses')
        return loss

    def add_entities_to_feed_dict(self, sess, feed_dict, batch, entity_type_map, type_entity_map,
                                  ep_e1_e2_map, e1_e2_ep_map, neg_samples, semi_hard=False):
        e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len = batch
        if self.type_id_map is None:
            self.type_id_map = {_t: i for i, _t in enumerate(type_entity_map.iterkeys())}
        feed_dict = super(RNNEntityWithTypes, self) \
            .add_entities_to_feed_dict(sess, feed_dict, batch, entity_type_map, type_entity_map,
                                       ep_e1_e2_map, e1_e2_ep_map, neg_samples, semi_hard)
        feed_dict[self.e1_type_batch] = [self.type_id_map[random.choice(entity_type_map[e])]
                                         if e in entity_type_map else self.type_vocab_size for e in e1]
        feed_dict[self.e2_type_batch] = [self.type_id_map[random.choice(entity_type_map[e])]
                                         if e in entity_type_map else self.type_vocab_size for e in e2]
        feed_dict[self.e1_neg_type_batch] = [random.randint(0, self.type_vocab_size - 1) for e in e1]
        feed_dict[self.e2_neg_type_batch] = [random.randint(0, self.type_vocab_size - 1) for e in e2]
        return feed_dict