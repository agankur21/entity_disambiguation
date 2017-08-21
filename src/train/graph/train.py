import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from src.evaluation.nearest_neighbors import NearestNeighbors
from src.evaluation.fb15k import *
from src.models.distmult import *
from src.utils.data_utils import *

FLAGS = tf.app.flags.FLAGS


def train_model(model, kb_batcher, sv, sess, saver, train_ops,
                evaluator,
                log_every, eval_every, neg_samples, save_path,
                max_decrease_epochs=5, max_steps=-1):
    step = 0.
    kb_examples, text_examples = 0., 0.
    kb_losses, text_losses, reconstruction_losses = [0], [0], [0]
    kb_loss_idx, text_loss_idx, reconstruction_loss_idx = 1, 1, 1
    loss_avg_len = FLAGS.text_batch * 500
    last_update = time.time()
    best_score = 0.0
    decrease_epochs = 0
    percentile = FLAGS.variance_min if FLAGS.variance_delta >= 0 else FLAGS.variance_max
    train_op, var_train_op, other_train_op = train_ops
    print ('Starting training')
    while not sv.should_stop() and (max_steps <= 0 or step < max_steps):
        var_weight = np.percentile(reconstruction_losses, int(min(percentile, 99.999))) if FLAGS.percentile \
            else (FLAGS.variance_min + (FLAGS.variance_delta * step))
        # kb update
        batch = kb_batcher.next_batch(sess)
        e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len = batch
        feed_dict = {model.kb_batch: rel, model.text_update: False, model.loss_weight: 1.0,
                     model.word_dropout_keep: FLAGS.word_dropout, model.lstm_dropout_keep: FLAGS.lstm_dropout,
                     model.final_dropout_keep: FLAGS.final_dropout, model.variance_weight: var_weight}
        feed_dict = model.add_entities_to_feed_dict(sess, feed_dict, batch, neg_samples, FLAGS.semi_hard)
        if var_train_op and FLAGS.alternate_var_train != 0:
            if FLAGS.alternate_var_train < 0 or int(step / FLAGS.alternate_var_train) % 2 == 1:
                if FLAGS.verbose: print('var_train_op')
                current_train_op = var_train_op
            else:
                if FLAGS.verbose: print('rest train op')
                current_train_op = other_train_op
        else:
            current_train_op = train_op
        _, global_step, loss, reconstruction_loss, _ = sess.run([current_train_op, model.global_step, model.loss,
                                                                 model.reconstruction_loss, model.renorm],
                                                                feed_dict=feed_dict)
        batch_size = e1.shape[0]
        loss /= batch_size
        # eval / serialize
        if step % eval_every == 0:
            if evaluator:
                results = evaluator.eval(block=(save_path is not None))
                if evaluator.eval_type == 'fb15K-237':
                    mrr, hits_at_10 = results
                    new_score = mrr + hits_at_10
                    print ('\n MRR: %2.3f \t hits@10: %2.3f \n' % (mrr, hits_at_10))
                else:
                    new_score = results
                # model is new best - serialize
                if new_score >= best_score:
                    decrease_epochs = 0
                    best_score = new_score
                    if save_path:
                        saved_path = saver.save(sess, save_path)
                        print("Serialized model: %s" % saved_path)
                # if model doesnt improve after max_decrease_epochs, stop training
                else:
                    decrease_epochs += 1
                    if decrease_epochs > max_decrease_epochs:
                        return
            elif save_path:
                saved_path = saver.save(sess, save_path)
                print("Serialized model: %s" % saved_path)

        kb_examples += batch_size
        if len(kb_losses) < loss_avg_len:
            kb_losses.append(loss)
            kb_loss_idx += 1
        else:
            kb_loss_idx = 0 if kb_loss_idx >= (loss_avg_len - 1) else kb_loss_idx + 1
            kb_losses[kb_loss_idx] = loss
        # log
        if step % log_every == 0:
            steps_per_sec = log_every / (time.time() - last_update)
            examples_per_sec = text_examples / (time.time() - last_update)
            text_examples = 0.
            sys.stdout.write('\rstep: %d \t kb loss: %.4f \t '
                             'text reconstruction loss: %.4f \t steps/sec: %.4f \t text examples/sec: %5.2f' %
                             (step, float(np.mean(kb_losses)), float(np.mean(reconstruction_losses)), steps_per_sec,
                              examples_per_sec))
            sys.stdout.flush()
            last_update = time.time()
        step += 1


def parse_vocabulary():
    """
    Parse the vocalubary of entities, relations and entity pair from the vocabulary directory
    :return:
    """
    with open(FLAGS.vocab_dir + '/rel.txt', 'r') as f:
        kb_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        kb_id_str_map = {i: s for s, i in kb_str_id_map.iteritems()}
        kb_vocab_size = len(kb_id_str_map)
    with open(FLAGS.vocab_dir + '/token.txt', 'r') as f:
        token_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        token_id_str_map = {i: s for s, i in token_str_id_map.iteritems()}
        token_vocab_size = len(token_id_str_map)
    with open(FLAGS.vocab_dir + '/entities.txt', 'r') as f:
        entity_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        entity_id_str_map = {i: s for s, i in entity_str_id_map.iteritems()}
        entity_vocab_size = len(entity_id_str_map)
    with open(FLAGS.vocab_dir + '/ep.txt', 'r') as f:
        ep_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        ep_id_str_map = {i: s for s, i in ep_str_id_map.iteritems()}
        ep_vocab_size = len(ep_id_str_map)
    e1_e2_ep_map = {(entity_str_id_map[ep_str.split('::')[0]], entity_str_id_map[ep_str.split('::')[1]]): ep_id
                    for ep_id, ep_str in ep_id_str_map.iteritems()}
    ep_e1_e2_map = {ep: e1_e2 for e1_e2, ep in e1_e2_ep_map.iteritems()}
    return kb_str_id_map, kb_id_str_map, kb_vocab_size, token_str_id_map, token_id_str_map, token_vocab_size, entity_str_id_map, entity_id_str_map, entity_vocab_size, ep_str_id_map, ep_id_str_map, ep_vocab_size, e1_e2_ep_map, ep_e1_e2_map


def main(argv):
    # print flags:values in alphabetical order
    print ('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    # Checking for 3 requirements : 1). Vocabulary 2).Indexed knowledge graph
    if FLAGS.vocab_dir == '':
        print('Error: Must supply input data generated from tsv_to_tfrecords.py')
        sys.exit(1)
    if FLAGS.kb_train == '' and FLAGS.text_train == '':
        print('Error: Must supply either kb_train or text_train')
        sys.exit(1)

    # Reading the dictionaries from vocabulary
    kb_str_id_map, kb_id_str_map, kb_vocab_size, token_str_id_map, token_id_str_map, token_vocab_size, entity_str_id_map, entity_id_str_map, entity_vocab_size, ep_str_id_map, ep_id_str_map, ep_vocab_size, e1_e2_ep_map, ep_e1_e2_map = parse_vocabulary()

    # Defining the graph
    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        random.seed(FLAGS.random_seed)

        # have seperate batchers for text and kb relation updates
        batcher = InMemoryGraphBatcher if FLAGS.in_memory else GraphBatcher
        kb_batcher = batcher(FLAGS.kb_train, FLAGS.kb_epochs, FLAGS.max_seq, FLAGS.kb_batch) \
            if FLAGS.kb_train != '' else None
        text_batcher = batcher(FLAGS.text_train, FLAGS.text_epochs, FLAGS.max_seq, FLAGS.text_batch) \
            if FLAGS.text_train != '' else None
        # construct the model
        model = DistMult(lr=FLAGS.lr, embed_dim=FLAGS.embed_dim, token_dim=FLAGS.token_dim, lstm_dim=FLAGS.lstm_dim,
                         entity_vocab_size=entity_vocab_size, kb_vocab_size=kb_vocab_size,
                         token_vocab_size=token_vocab_size,
                         loss_type=FLAGS.loss_type, margin=FLAGS.margin, l2_weight=FLAGS.l2_weight,
                         neg_samples=FLAGS.neg_samples,
                         norm_entities=FLAGS.norm_entities, use_tanh=FLAGS.use_tanh, max_pool=FLAGS.max_pool,
                         bidirectional=FLAGS.bidirectional, peephole=FLAGS.use_peephole,
                         verbose=FLAGS.verbose, freeze=FLAGS.freeze)

        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, epsilon=FLAGS.epsilon) if FLAGS.optimizer == 'adam' \
            else tf.train.AdagradOptimizer(learning_rate=FLAGS.lr)
        if FLAGS.clip_norm > 0:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), FLAGS.clip_norm)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)
            variance_vars = [k for k in tvars if 'variance' in k.name]
            if variance_vars:
                var_grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, variance_vars), FLAGS.clip_norm)
                var_train_op = optimizer.apply_gradients(zip(var_grads, variance_vars), global_step=model.global_step)
            else:
                var_train_op = None
            other_vars = [k for k in tvars if 'variance' not in k.name]
            other_grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, other_vars), FLAGS.clip_norm)
            other_train_op = optimizer.apply_gradients(zip(other_grads, other_vars), global_step=model.global_step)
            train_ops = [train_op, var_train_op, other_train_op]
        else:
            train_ops = [optimizer.minimize(model.loss, global_step=model.global_step), None, None]

        # restore only variables that exist in the checkpoint - needed to pre-train big models with small models
        if FLAGS.load_model != '':
            reader = tf.train.NewCheckpointReader(FLAGS.load_model)
            cp_list = set([key for key in reader.get_variable_to_shape_map()])
            # if variable does not exist in checkpoint or sizes do not match, dont load
            r_vars = [k for k in tf.global_variables() if k.name.split(':')[0] in cp_list
                      and k.get_shape() == reader.get_variable_to_shape_map()[k.name.split(':')[0]]]
            if len(cp_list) != len(r_vars):
                print('[Warning]: not all variables loaded from file')
            saver = tf.train.Saver(var_list=r_vars)
        else:
            saver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir=FLAGS.logdir,
                                 global_step=model.global_step,
                                 saver=None,
                                 save_summaries_secs=0,
                                 save_model_secs=0, )

        with sv.managed_session(FLAGS.master,
                                config=tf.ConfigProto(
                                    allow_soft_placement=True
                                )) as sess:
            if FLAGS.load_model != '':
                print("Deserializing model: %s" % FLAGS.load_model)
                saver.restore(sess, FLAGS.load_model)
            evaluator = FB15KEvaluator(model, fb15k_dir=FLAGS.fb15k_dir, session=sess,
                                       kb_str_id_map=kb_str_id_map, kb_id_str_map=kb_id_str_map,
                                       entity_str_id_map=entity_str_id_map,
                                       entity_id_str_map=entity_id_str_map,
                                       ep_str_id_map=ep_str_id_map, ep_id_str_map=ep_id_str_map)
            threads = tf.train.start_queue_runners(sess=sess)
            if FLAGS.mode == 'train':
                if FLAGS.in_memory:
                    if text_batcher: text_batcher.load_all_data(sess)
                    if kb_batcher: kb_batcher.load_all_data(sess)
                save_path = '%s/%s' % (FLAGS.logdir, FLAGS.save_model) if FLAGS.save_model != '' else None
                train_model(model=model, kb_batcher=kb_batcher, sv=sv, sess=sess, saver=saver, train_ops=train_ops,
                            evaluator=evaluator, log_every=FLAGS.log_every, eval_every=FLAGS.eval_every,
                            neg_samples=FLAGS.neg_samples,
                            save_path=save_path, max_decrease_epochs=FLAGS.max_decrease_epochs,
                            max_steps=FLAGS.max_steps)
            elif FLAGS.mode == 'evaluate':
                print('Evaluating')
                results = evaluator.eval(block=True)
                print (results)
            elif 'similarity' in FLAGS.mode:
                nn = NearestNeighbors(model, session=sess,
                                      kb_str_id_map=kb_str_id_map, kb_id_str_map=kb_id_str_map,
                                      ep_str_id_map=ep_str_id_map, ep_id_str_map=ep_id_str_map,
                                      entity_str_id_map=entity_str_id_map, entity_id_str_map=entity_id_str_map)
                if FLAGS.mode == 'entity_similarity':
                    nn.entity_similarity()
                elif FLAGS.mode == 'kb_similarity':
                    nn.kb_similarity()
            else:
                print('Error: "%s" is not a valid mode' % FLAGS.mode)
                sys.exit(1)
            sv.coord.request_stop()
            sv.coord.join(threads)
            sess.close()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('vocab_dir', '', 'tsv file containing string data')
    tf.app.flags.DEFINE_string('kb_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('text_train', '',
                               'file pattern of proto buffers generated from ../src/processing/tsv_to_tfrecords.py')
    tf.app.flags.DEFINE_string('fb15k_dir', '', 'directory containing fb15k tsv files')
    tf.app.flags.DEFINE_string('nci_dir', '', 'directory containing nci tsv files')
    tf.app.flags.DEFINE_string('noise_dir', '',
                               'directory containing fb15k noise files generated from src/util/generate_noise.py')
    tf.app.flags.DEFINE_string('candidate_file', '', 'candidate file for tac evaluation')
    tf.app.flags.DEFINE_string('variance_file', '', 'variance file in candidate file format')
    tf.app.flags.DEFINE_string('type_file', '', 'tsv mapping entities to types')
    tf.app.flags.DEFINE_string('logdir', '', 'save logs and models to this dir')
    tf.app.flags.DEFINE_string('load_model', '', 'path to saved model to load')
    tf.app.flags.DEFINE_string('save_model', '', 'name of file to serialize model to')
    tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use')
    tf.app.flags.DEFINE_string('loss_type', 'softmax', 'optimizer to use')
    tf.app.flags.DEFINE_string('model_type', 'd', 'optimizer to use')
    tf.app.flags.DEFINE_string('text_encoder', 'lstm', 'optimizer to use')
    tf.app.flags.DEFINE_string('variance_type', 'divide', 'type of variance model to use')
    tf.app.flags.DEFINE_string('mode', 'train', 'train, evaluate, analyze')
    tf.app.flags.DEFINE_string('master', '', 'use for Supervisor')

    tf.app.flags.DEFINE_boolean('norm_entities', False, 'normalize entitiy vectors to have unit norm')
    tf.app.flags.DEFINE_boolean('bidirectional', False, 'bidirectional lstm')
    tf.app.flags.DEFINE_boolean('use_tanh', False, 'use tanh')
    tf.app.flags.DEFINE_boolean('use_peephole', False, 'use peephole connections in lstm')
    tf.app.flags.DEFINE_boolean('max_pool', False, 'max pool hidden states of lstm, else take last')
    tf.app.flags.DEFINE_boolean('in_memory', False, 'load data in memory')
    tf.app.flags.DEFINE_boolean('reset_variance', False, 'reset loaded variance projection matrices')
    tf.app.flags.DEFINE_boolean('percentile', False, 'variance weight based off of percentile')
    tf.app.flags.DEFINE_boolean('semi_hard', False, 'use semi hard negative sample selection')
    tf.app.flags.DEFINE_boolean('verbose', False, 'additional logging')
    tf.app.flags.DEFINE_boolean('freeze', False, 'freeze row and column params')

    # tac eval
    tf.app.flags.DEFINE_boolean('center_only', False, 'only take center in tac eval')
    tf.app.flags.DEFINE_boolean('arg_entities', False, 'replaced entities with arg wildcards')
    tf.app.flags.DEFINE_boolean('norm_digits', False, 'norm digits in tac eval')

    tf.app.flags.DEFINE_float('lr', .01, 'learning rate')
    tf.app.flags.DEFINE_float('epsilon', 1e-8, 'epsilon for adam optimizer')
    tf.app.flags.DEFINE_float('margin', 1.0, 'margin for hinge loss')
    tf.app.flags.DEFINE_float('l2_weight', 1.0, 'weight for l2 loss')
    tf.app.flags.DEFINE_float('clip_norm', 1, 'clip gradients to have norm <= this')
    tf.app.flags.DEFINE_float('text_weight', .25, 'weight for text updates')
    tf.app.flags.DEFINE_float('text_prob', .5, 'probability of drawing a text batch vs kb batch')
    tf.app.flags.DEFINE_float('variance_min', 1.0, 'weight of variance penalty')
    tf.app.flags.DEFINE_float('variance_max', 1.0, 'weight of variance penalty')
    tf.app.flags.DEFINE_float('variance_delta', .0001, 'increase variance weight by this value each step')
    tf.app.flags.DEFINE_float('word_dropout', .9, 'dropout keep probability for word embeddings')
    tf.app.flags.DEFINE_float('lstm_dropout', 1.0, 'dropout keep probability for lstm output before projection')
    tf.app.flags.DEFINE_float('final_dropout', 1.0, 'dropout keep probability for final row and column representations')

    tf.app.flags.DEFINE_integer('kb_vocab_size', 237, 'learning rate')
    tf.app.flags.DEFINE_integer('text_batch', 128, 'batch size')
    tf.app.flags.DEFINE_integer('kb_batch', 4096, 'batch size')
    tf.app.flags.DEFINE_integer('token_dim', 250, 'token dimension')
    tf.app.flags.DEFINE_integer('lstm_dim', 2048, 'lstm internal dimension')
    tf.app.flags.DEFINE_integer('embed_dim', 100, 'row/col embedding dimension')
    tf.app.flags.DEFINE_integer('position_dim', 5, 'position relative to entities in lstm embedding')
    tf.app.flags.DEFINE_integer('text_epochs', 100, 'train for this many text epochs')
    tf.app.flags.DEFINE_integer('kb_epochs', 100, 'train for this many kb epochs')
    tf.app.flags.DEFINE_integer('kb_pretrain', 0, 'pretrain kb examples for this many steps')
    tf.app.flags.DEFINE_integer('alternate_var_train', 0,
                                'alternate between variance and rest optimizers every k steps')
    tf.app.flags.DEFINE_integer('log_every', 20, 'log every k steps')
    tf.app.flags.DEFINE_integer('eval_every', 30, 'eval every k steps')
    tf.app.flags.DEFINE_integer('max_steps', -1, 'stop training after this many total steps')
    tf.app.flags.DEFINE_integer('max_seq', 15, 'maximum sequence length')
    tf.app.flags.DEFINE_integer('max_decrease_epochs', 10, 'stop training early if eval doesnt go up')
    tf.app.flags.DEFINE_integer('neg_samples', 200, 'number of negative samples')
    tf.app.flags.DEFINE_integer('random_seed', 1111, 'random seed')

tf.app.run()
