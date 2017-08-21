from src.models.text_encoders import *


class JointContextModel(object):
    def __init__(self, token_size, token_dim, mention_size, mention_dim, entity_size, entity_dim, lstm_dim,
                 embed_dim, final_out_dim, word_dropout_keep,
                 lstm_dropout_keep, final_dropout_keep, non_linearity, threshold=0.5, l2_weight=0.0005, embeddings=None,
                 freeze=False):
        # Defining model variables
        self.lstm_dropout_keep = lstm_dropout_keep
        self.final_out_dim = final_out_dim
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.final_dropout_keep = final_dropout_keep
        self.non_linearity = non_linearity
        self.l2_weight = l2_weight
        self.threshold = tf.constant(threshold, dtype=tf.float32)
        # Defining Placeholder
        self.seq_len_left = tf.placeholder_with_default([0], [None], name='seq_len_left')
        self.seq_len_right = tf.placeholder_with_default([0], [None], name='seq_len_right')
        self.left_tokens = tf.placeholder_with_default([[0]], [None, None], name='left_tokens')
        self.right_tokens = tf.placeholder_with_default([[0]], [None, None], name='right_tokens')
        self.mentions = tf.placeholder_with_default([[0]], [None, None], name='mentions')
        self.entities = tf.placeholder_with_default([[0]], [None, None], name='entities')
        self.labels = tf.placeholder_with_default([[0.0]], [None, 1], name='labels')

        # Defining Variables for local context embedding
        self.token_embeddings = tf_utils.initialize_embeddings(shape=[token_size - 1, token_dim],
                                                               name="token_embeddings", pretrained=embeddings,
                                                               freeze=(not freeze))
        self.local_context_projection = tf.get_variable(name='local_context_projection',
                                                        shape=[2 * embed_dim, embed_dim],
                                                        initializer=tf.contrib.layers.xavier_initializer(),
                                                        trainable=(not freeze))
        self.local_context_bias = tf.get_variable(name='local_context_bias',
                                                  initializer=tf.constant(0.0001, shape=[embed_dim]))
        self.left_lstm_text_encoder = LSTMTextEncoder(text_placeholder=self.left_tokens,
                                                      seq_len_placeholder=self.seq_len_left,
                                                      lstm_dim=lstm_dim, embed_dim=embed_dim, token_dim=token_dim,
                                                      word_dropout_keep=word_dropout_keep,
                                                      lstm_dropout_keep=lstm_dropout_keep,
                                                      final_dropout_keep=final_dropout_keep, use_birectional=False)
        self.right_lstm_text_encoder = LSTMTextEncoder(text_placeholder=self.right_tokens,
                                                       seq_len_placeholder=self.seq_len_right,
                                                       lstm_dim=lstm_dim, embed_dim=embed_dim, token_dim=token_dim,
                                                       word_dropout_keep=word_dropout_keep,
                                                       lstm_dropout_keep=lstm_dropout_keep,
                                                       final_dropout_keep=final_dropout_keep, use_birectional=False)
        # Defining variable for global context
        self.global_context_projection = tf_utils.initialize_embeddings(shape=[mention_size - 1, mention_dim],
                                                                        name="mention_embeddings", pretrained=None,
                                                                        freeze=(not freeze))
        self.global_context_bias = tf.get_variable(name='global_context_bias',
                                                   initializer=tf.constant(0.0001, shape=[mention_dim]))
        # Defining variables for entity labels
        self.entity_embeddings = tf_utils.initialize_embeddings(shape=[entity_size - 1, entity_dim],
                                                                name="entity_embeddings", pretrained=None,
                                                                freeze=(not freeze))
        self.entity_embeddings_bias = tf.get_variable(name='entity_embeddings_bias',
                                                      initializer=tf.constant(0.0001, shape=[entity_dim]))
        # Defining final layer weight and bias
        self.final_projection = tf.get_variable(name='final_projection',
                                                shape=[final_out_dim, 1],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                trainable=(not freeze))
        self.final_bias = tf.get_variable(name='final_bias', initializer=tf.constant(0.0001, shape=[1]))

        # Defining Loss and Prediction
        self.logits = self.forward()
        self.loss = self.calculate_loss()
        self.predictions = self.calculate_prediction()

    def get_context_embedding(self, scope='joint_context', reuse=False):
        """
        Get the context embedding of a mention
        :param scope:
        :param reuse:
        :return:
        """
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope("local_context", reuse=reuse):
                with tf.variable_scope('left', reuse=reuse):
                    left_sentence_out = self.left_lstm_text_encoder.embed_text(
                        token_embeddings=self.token_embeddings, reuse=reuse)
                with tf.variable_scope('right', reuse=reuse):
                    right_sentence_out = self.right_lstm_text_encoder.embed_text(
                        token_embeddings=self.token_embeddings, reuse=reuse)
                local_context_projections = tf_utils.concat_and_project([left_sentence_out, right_sentence_out],
                                                                        self.local_context_projection,
                                                                        self.local_context_bias, self.lstm_dropout_keep)
                local_context_projected_non_linear = tf_utils.apply_nonlinearity(local_context_projections,
                                                                                 nonlinearity_type=self.non_linearity)
            with tf.variable_scope("global_context", reuse=reuse):
                mask = tf.expand_dims(tf.cast(tf.sign(self.mentions), tf.float32), 2)
                mask_count = tf.reduce_sum(mask, axis=1)
                mention_embeddings = tf.nn.embedding_lookup(self.global_context_projection, self.mentions)
                global_context_projections = tf.div(tf.reduce_sum(mention_embeddings, axis=1),
                                                    tf.maximum(mask_count, tf.ones_like(mask_count)))
                global_context_projections = tf.nn.bias_add(global_context_projections, self.global_context_bias)
                global_context_projected_non_linear = tf_utils.apply_nonlinearity(global_context_projections,
                                                                                  nonlinearity_type=self.non_linearity)
        return tf.concat(axis=1, values=[local_context_projected_non_linear, global_context_projected_non_linear])

    def get_entity_embedding(self, reuse=False):
        with tf.variable_scope("entity_embeddings", reuse=reuse):
            mask = tf.expand_dims(tf.cast(tf.sign(self.entities), tf.float32), 2)
            mask_count = tf.reduce_sum(mask, axis=1)
            entity_embeddings = tf.nn.embedding_lookup(self.entity_embeddings, self.entities)
            entity_projections = tf.div(tf.reduce_sum(entity_embeddings, axis=1),
                                        tf.maximum(mask_count, tf.ones_like(mask_count)))
            entity_projections = tf.nn.bias_add(entity_projections, self.entity_embeddings_bias)
            entity_projected_non_linear = tf_utils.apply_nonlinearity(entity_projections,
                                                                      nonlinearity_type=self.non_linearity)
        return entity_projected_non_linear

    def forward(self, scope='forward', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            context_embedding = self.get_context_embedding(reuse=reuse)
            entity_embeddings = self.get_entity_embedding(reuse=reuse)
            with tf.variable_scope("DenseLayers-Positive", reuse=reuse):
                dense1 = tf.layers.dense(inputs=tf.concat([context_embedding, entity_embeddings], axis=1),
                                         units=self.final_out_dim,
                                         activation=tf_utils.get_non_linear_activation(self.non_linearity),
                                         name='dense-1')
                sparse1 = tf.nn.dropout(x=dense1, keep_prob=self.final_dropout_keep, name="sparse-1")
                logits = tf.nn.xw_plus_b(x=sparse1, weights=self.final_projection, biases=self.final_bias,
                                         name="sigmoid")
        return logits

    def calculate_loss(self):
        logits = self.logits
        softmax_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits))
        l2_loss = self.l2_weight * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if
                                       not ("noreg" in tf_var.name or "bias" in tf_var.name or "lstm" in tf_var.name))
        return softmax_loss + l2_loss

    def calculate_prediction(self, reuse=False):
        with tf.variable_scope("Prediction", reuse=reuse):
            logits = self.logits
            sigmoids = tf.sigmoid(logits)  # for prediction
            return tf.cast(tf.greater(sigmoids, self.threshold), tf.float32)
