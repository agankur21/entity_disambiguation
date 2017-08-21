import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from config.data import *
from src.evaluation.classifier import BinaryClassifierEvaluator
from src.models.joint_context_model import *
from src.utils.data_utils import *
from src.utils.tf_utils import *

##################################################################################################
tf.app.flags.DEFINE_string('vocab_dir', os.path.join(DATA_DIR, "ncbi_disease_corpus", "vocab"),
                           'tsv file containing string data')
tf.app.flags.DEFINE_string('text_in_files', os.path.join(TRAINING_DATA_PATH, "Corpus.txt"),
                           'input text file in pubtator format')
tf.app.flags.DEFINE_string('text_train', os.path.join(DATA_DIR, "ncbi_disease_corpus", "vocab", "train.proto"),
                           'file pattern of proto buffers generated from ../src/preprocessing/mentions/tsv_to_tfrecords.py')
tf.app.flags.DEFINE_string('text_dev', '',
                           'file pattern of proto buffers generated from ../src/preprocessing/mentions/tsv_to_tfrecords.py')
tf.app.flags.DEFINE_string('logdir', os.path.join(DATA_DIR, "ncbi_disease_corpus", "saved_models"),
                           'save logs and models to this dir')
tf.app.flags.DEFINE_string('load_model', '', 'path to saved model to load')
tf.app.flags.DEFINE_string('save_model', 'model.tf', 'name of file to serialize model to')

tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use')
tf.app.flags.DEFINE_string('loss_type', 'softmax', 'optimizer to use')
tf.app.flags.DEFINE_string('text_encoder', 'lstm', 'optimizer to use')

tf.app.flags.DEFINE_string('mode', 'train', 'train, evaluate, analyze')
tf.app.flags.DEFINE_string('master', '', 'use for Supervisor')

tf.app.flags.DEFINE_boolean('use_tanh', False, 'use tanh')
tf.app.flags.DEFINE_string('non_linearity', 'tanh', 'Non linearity to be used')

tf.app.flags.DEFINE_boolean('verbose', False, 'additional logging')
tf.app.flags.DEFINE_boolean('freeze', False, 'freeze row and column params')

tf.app.flags.DEFINE_float('lr', .001, 'learning rate')
tf.app.flags.DEFINE_float('epsilon', 1e-8, 'epsilon for adam optimizer')
tf.app.flags.DEFINE_float('margin', 1.0, 'margin for hinge loss')
tf.app.flags.DEFINE_float('l2_weight', 0.005, 'weight for l2 loss')
tf.app.flags.DEFINE_float('clip_norm', 1, 'clip gradients to have norm <= this')

tf.app.flags.DEFINE_float('word_dropout', .9, 'dropout keep probability for word embeddings')
tf.app.flags.DEFINE_float('lstm_dropout', 1.0, 'dropout keep probability for lstm output before projection')
tf.app.flags.DEFINE_float('final_dropout', 1.0, 'dropout keep probability for final row and column representations')

tf.app.flags.DEFINE_integer('text_batch', 64, 'batch size')
tf.app.flags.DEFINE_integer('token_dim', 100, 'token dimension')
tf.app.flags.DEFINE_integer('mention_dim', 100, 'token dimension')
tf.app.flags.DEFINE_integer('entity_dim', 100, 'entity embedding dimension')
tf.app.flags.DEFINE_integer('lstm_dim', 1024, 'lstm internal dimension')
tf.app.flags.DEFINE_integer('embed_dim', 100, 'row/col embedding dimension')
tf.app.flags.DEFINE_integer('final_out_dim', 100, 'row/col embedding dimension')
tf.app.flags.DEFINE_integer('max_seq', 30, 'maximum sequence length')
tf.app.flags.DEFINE_integer('text_epochs', 10, 'train for this many text epochs')
tf.app.flags.DEFINE_integer('eval_every', 10, 'eval every k steps')
tf.app.flags.DEFINE_integer('max_decrease_epochs', 5, 'stop training early if eval doesnt go up')
tf.app.flags.DEFINE_integer('neg_samples', 10, 'number of negative samples')
tf.app.flags.DEFINE_integer('random_seed', 1111, 'random seed')
tf.app.flags.DEFINE_string('embeddings',
                           '/Users/aaggarwal/Documents/Course/CZI/EntityLinking/entity_linking/data/embeddings/lample-embeddings-pre.txt',
                           'file of pretrained embeddings to use')
tf.app.flags.DEFINE_float('threshold', '0.1', 'threshold for choosing the label')

FLAGS = tf.app.flags.FLAGS


###################################################################################################

def load_word_embeddings(vocab_str_id_map):
    """
    Using the vocab map and the pretrained embeddings if exists, load the word embeddings
    :param vocab_str_id_map:
    :return:
    """
    # load embeddings, if given; initialize in range [-.01, .01]
    vocab_size = len(vocab_str_id_map)
    # Since the vocab contains an empty string as well we need to subtract it from the embedding
    embeddings_shape = (vocab_size - 1, FLAGS.token_dim)
    embeddings = tf_utils.embedding_values(embeddings_shape, old=False)
    # In case of pretrained embeddings
    word_set_used = set([])
    if FLAGS.embeddings != '':
        print "Loading the pre-trained word embedding file : %s" % FLAGS.embeddings
        with open(FLAGS.embeddings, 'r') as f:
            for line in f.readlines():
                split_line = line.strip().split(" ")
                word = split_line[0]
                embedding = split_line[1:]
                if word in vocab_str_id_map:
                    # shift by -1 because we are going to add a 0 constant vector for the padding later
                    embeddings[vocab_str_id_map[word] - 1] = map(float, embedding)
                    word_set_used.add(word)
                elif word.lower() in vocab_str_id_map:
                    embeddings[vocab_str_id_map[word.lower()] - 1] = map(float, embedding)
                    if word.lower() not in word_set_used:
                        word_set_used.add(word.lower())

    print("Loaded %d/%d embeddings (%2.2f%% coverage)" % (
        len(word_set_used), vocab_size, 1.0 * len(word_set_used) / vocab_size * 100))
    return embeddings


def load_vocab(vocab_dir):
    """
    Loading the vocabulary already saved from the folder
    :return:
    """
    # If the vocabulary is already created, load it
    print('Loading vocab from %s' % vocab_dir)
    with open('%s/entities.txt' % vocab_dir) as f:
        entity_map = {l.split('\t')[0]: int(l.split('\t')[1]) for l in f}
    with open('%s/token.txt' % vocab_dir) as f:
        token_map = {l.split('\t')[0]: int(l.split('\t')[1]) for l in f}
    with open('%s/mention.txt' % vocab_dir) as f:
        mention_map = {l.split('\t')[0]: int(l.split('\t')[1]) for l in f}
    print('Loaded %d tokens, %d entities %d mentions'
          % (len(token_map), len(entity_map), len(mention_map)))
    return entity_map, token_map, mention_map


def train_model(model, train_op, batcher, dev_batches, sv, sess, saver, max_entity_size, num_neg_samples, save_path,
                eval_every, evaluator, max_decrease_epochs=15, max_steps=-1):
    step = 0.0
    best_score = 0.0
    decrease_epochs = 0
    print ('Starting training')
    while not sv.should_stop() and (max_steps <= 0 or step < max_steps):
        print ("Executing Training Step :%d" % (step + 1))
        batch = sess.run(batcher.next_batch_op)
        seq_len_left_batch, seq_len_right_batch, left_tokens_batch, right_tokens_batch, mentions_batch, entities_batch = batch
        batch_size = entities_batch.shape[0]
        # Extending the batch with negative examples
        print ("Adding %d negative samples for each positive example" % num_neg_samples)
        seq_len_left_batch = extend_batch(seq_len_left_batch, num_neg_samples)
        seq_len_right_batch = extend_batch(seq_len_right_batch, num_neg_samples)
        left_tokens_batch = extend_batch(left_tokens_batch, num_neg_samples)
        right_tokens_batch = extend_batch(right_tokens_batch, num_neg_samples)
        mentions_batch = extend_batch(mentions_batch, num_neg_samples)
        entities_batch = generate_negatives(entities_batch, max_entity_size, num_neg_samples)
        labels_batch = get_labels(batch_size, num_neg_samples)
        feed_dict = {model.seq_len_left: seq_len_left_batch, model.seq_len_right: seq_len_right_batch,
                     model.left_tokens: left_tokens_batch, model.right_tokens: right_tokens_batch,
                     model.mentions: mentions_batch, model.entities: entities_batch,
                     model.labels: labels_batch}
        _, global_step, loss, predictions = sess.run([train_op, model.global_step, model.loss, model.predictions],
                                                     feed_dict=feed_dict)
        print "Loss: %f" % loss
        evaluator.print_output_score(predictions, labels_batch)
        loss /= batch_size
        # eval / serialize
        if (step + 1) % eval_every == 0:
            if evaluator:
                results = evaluator.eval(dev_batches)
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
        step += 1


def get_all_batches(sess, batcher, max_entity_size, num_neg_samples=10):
    batches = []
    # load all the dev batches into memory
    done = False
    num_examples = 0
    while not done:
        try:
            batch = sess.run(batcher.next_batch_op)
            seq_len_left_batch, seq_len_right_batch, left_tokens_batch, right_tokens_batch, mentions_batch, entities_batch = batch
            batch_size = entities_batch.shape[0]
            # Extending the batch with negative examples
            seq_len_left_batch = extend_batch(seq_len_left_batch, num_neg_samples)
            seq_len_right_batch = extend_batch(seq_len_right_batch, num_neg_samples)
            left_tokens_batch = extend_batch(left_tokens_batch, num_neg_samples)
            right_tokens_batch = extend_batch(right_tokens_batch, num_neg_samples)
            mentions_batch = extend_batch(mentions_batch, num_neg_samples)
            entities_batch = generate_negatives(entities_batch, max_entity_size, num_neg_samples)
            labels_batch = get_labels(batch_size, num_neg_samples)
            num_examples += labels_batch.shape[0]
            batches.append((seq_len_left_batch, seq_len_right_batch, left_tokens_batch, right_tokens_batch,
                            mentions_batch, entities_batch, labels_batch))
        except:
            done = True
    print "Statistics of Dev batch: Number of Batches: %d" % (len(batches))
    print "Total number of examples : %d" % num_examples
    print "Negative Samples per Positive example in Dev Batch: %d" % num_neg_samples
    return batches


def main(argv):
    # print flags:values in alphabetical order
    print ('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    # Checking for 3 requirements : 1). Vocabulary 2).Indexed knowledge graph
    if FLAGS.vocab_dir == '':
        print('Error: Must supply input data generated from tsv_to_tfrecords.py')
        sys.exit(1)
    if FLAGS.text_train == '':
        print('Error: Must supply text_train')
        sys.exit(1)
    if FLAGS.text_in_files == '':
        print('Error: Must supply text files for validation')
        sys.exit(1)
    # Reading the dictionaries from vocabulary
    if FLAGS.text_dev == "":
        FLAGS.text_dev = os.path.join(os.path.dirname(FLAGS.text_train), "dev.proto")
    entity_map, token_map, mention_map = load_vocab(FLAGS.vocab_dir)

    # Initialize word embeddings and update them with pretrained ones if exists
    embeddings = load_word_embeddings(token_map)

    # Defining the graph
    with tf.Graph().as_default():
        # Fixing the seed
        tf.set_random_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        random.seed(FLAGS.random_seed)

        # Batcher for input batches
        train_batcher = MentionBatcher(FLAGS.text_train, FLAGS.text_epochs, FLAGS.max_seq, FLAGS.text_batch)
        dev_batcher = MentionBatcher(FLAGS.text_dev, 1, FLAGS.max_seq, FLAGS.text_batch)
        # construct the model
        model = JointContextModel(token_size=len(token_map), token_dim=FLAGS.token_dim, mention_size=len(mention_map),
                                  mention_dim=FLAGS.mention_dim,
                                  entity_size=len(entity_map), entity_dim=FLAGS.entity_dim, lstm_dim=FLAGS.lstm_dim,
                                  embed_dim=FLAGS.embed_dim, final_out_dim=FLAGS.final_out_dim,
                                  word_dropout_keep=FLAGS.word_dropout, lstm_dropout_keep=FLAGS.lstm_dropout,
                                  final_dropout_keep=FLAGS.final_dropout, embeddings=embeddings,
                                  non_linearity=FLAGS.non_linearity, threshold=FLAGS.threshold)
        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, epsilon=FLAGS.epsilon)

        # Defining train ops
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), FLAGS.clip_norm)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)

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
            evaluator = BinaryClassifierEvaluator(model, session=sess, entity_map=entity_map,
                                                  token_map=token_map, mention_map=mention_map,
                                                  threshold=FLAGS.threshold)
            threads = tf.train.start_queue_runners(sess=sess)
            dev_batches = get_all_batches(sess, dev_batcher, max_entity_size=len(entity_map))
            if FLAGS.mode == 'train':
                save_path = '%s/%s' % (FLAGS.logdir, FLAGS.save_model) if FLAGS.save_model != '' else None
                train_model(model=model, train_op=train_op, batcher=train_batcher, dev_batches=dev_batches, sv=sv,
                            sess=sess, saver=saver, max_entity_size=len(entity_map), num_neg_samples=FLAGS.neg_samples,
                            save_path=save_path, max_decrease_epochs=FLAGS.max_decrease_epochs, max_steps=-1,
                            evaluator=evaluator, eval_every=FLAGS.eval_every)
            elif FLAGS.mode == 'evaluate':
                print('Evaluating')
                results = evaluator.eval(dev_batches)
                print (results)
            else:
                print('Error: "%s" is not a valid mode' % FLAGS.mode)
                sys.exit(1)
            sv.coord.request_stop()
            sv.coord.join(threads)
            sess.close()


if __name__ == '__main__':
    tf.app.run()
