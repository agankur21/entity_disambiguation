from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import tensorflow as tf
import operator
import glob
import gzip
from collections import defaultdict
import multiprocessing
from functools import partial

########################Defining input flags #########################################
tf.app.flags.DEFINE_string('kg_in_files', '', 'pattern to match kg input files')
tf.app.flags.DEFINE_string('text_in_files', '', 'pattern to match text input files')
tf.app.flags.DEFINE_string('out_dir', '', 'export tf protos')
tf.app.flags.DEFINE_string('load_vocab', '', 'directory containing vocab files to load')
tf.app.flags.DEFINE_integer('max_len', 20, 'maximum sequence length')
tf.app.flags.DEFINE_integer('min_count', 5, 'replace tokens occuring less than this many times with <UNK>')
tf.app.flags.DEFINE_integer('num_threads', 5, 'max number of threads to use for parallel processing')
tf.app.flags.DEFINE_boolean('padding', 0, '0: no padding, 1: 0 pad to the right of seq, 2: 0 pad to the left')
tf.app.flags.DEFINE_boolean('normalize_digits', False, 'map all digits to 0')
tf.app.flags.DEFINE_boolean('tsv_format', False, 'data in 13 column tsv format')
FLAGS = tf.app.flags.FLAGS

################## Helper functions for creating Example objects################################
feature = tf.train.Feature
sequence_example = tf.train.SequenceExample


def features(d): return tf.train.Features(feature=d)


def int64_feature(v): return feature(int64_list=tf.train.Int64List(value=v))


def feature_list(l): return tf.train.FeatureList(feature=l)


def feature_lists(d): return tf.train.FeatureLists(feature_list=d)


queue = multiprocessing.Queue()
queue.put(0)
lock = multiprocessing.Lock()


##################################################################################################

def update_vocab_counts(line, entity_counter, entity_pair_counter, rel_map, token_counter, update_rels):
    """
    Parse each line and update the entity counter with number of times each entity is encountered,entity pair counter
    number of times each entity pair is encountered, rel_map with the index if new relation is found and token count of
    each different token in the relation string
    :param line:
    :param entity_counter:
    :param entity_pair_counter:
    :param rel_map:
    :param token_counter:
    :param update_rels:
    :return:
    """
    parts = line.strip().split('\t')
    if FLAGS.tsv_format:
        if len(parts) != 13:
            print('\nIncorrect number of fields in line: %s ' % line)
            return 1
        e1_str, e1_type, e1_mention_str, e1_s, e1_e, \
        e2_str, e2_type, e2_mention_str, e2_s, e2_e, \
        doc_id, doc_info, rel_str = parts
    else:
        if len(parts) != 4:
            print('\nIncorrect number of fields in line: %s ' % line)
            return 1
        e1_str, e2_str, rel_str, _ = parts

    # Normalize the digits to all be 0
    rel_str_normalized = re.sub(r'(?<!\$ARG)[0-9]', '0', rel_str) if FLAGS.normalize_digits else rel_str
    tokens_str = rel_str_normalized.split(' ')
    if len(tokens_str) <= FLAGS.max_len:
        ep_str = e1_str + '::' + e2_str
        # memory map strings -> ints
        entity_counter[e1_str] += 1
        entity_counter[e2_str] += 1
        entity_pair_counter[ep_str] += 1
        if update_rels and rel_str not in rel_map:
            rel_map[rel_str] = len(rel_map)
        for token in tokens_str:
            token_counter[token] += 1
    return 0


def make_example(entity_map, entity_pair_map, relation_map, token_map, line, writer):
    """
    Convert each line into indices
    :param entity_map:
    :param entity_pair_map:
    :param relation_map:
    :param token_map:
    :param line:
    :param writer:
    :return:
    """
    parts = line.strip().split('\t')
    if FLAGS.tsv_format:
        if len(parts) != 13:
            print('\nIncorrect number of fields in line: %s ' % line)
            return 0
        e1_str, e1_type, e1_mention_str, e1_s_str, e1_e_str, \
        e2_str, e2_type, e2_mention_str, e2_s_str, e2_e_str, \
        doc_id, doc_info, rel_str = parts
        e1_start = int(e1_s_str)
        e1_end = int(e1_e_str)
        e2_start = int(e2_s_str)
        e2_end = int(e2_e_str)
        # normalize the digits to all be 0
        rel_str_normalized = re.sub(r'(?<!\$ARG)[0-9]', '0', rel_str) \
            if FLAGS.normalize_digits else rel_str
        tokens_str = rel_str_normalized.split(' ')
    else:
        if len(parts) != 4:
            print('\nIncorrect number of fields in line: %s ' % line)
            return 0
        e1_str, e2_str, rel_str, _ = line.strip().split('\t')  # data format is 'e1 \t rel \t e2'
        # normalize the digits to all be 0
        rel_str_normalized = re.sub(r'(?<!\$ARG)[0-9]', '0', rel_str) if FLAGS.normalize_digits else rel_str
        tokens_str = rel_str_normalized.split(' ')
        e1_start = tokens_str.index('$ARG1') if '$ARG1' in tokens_str else 0
        e1_end = e1_start + 1
        e2_start = tokens_str.index('$ARG2') if '$ARG2' in tokens_str else 1
        e2_end = e2_start + 1

    if len(tokens_str) <= FLAGS.max_len:
        ep_str = e1_str + '::' + e2_str
        if e1_str not in entity_map:
            print('Warning %s not in entity vocab map' % e1_str)
        elif e2_str not in entity_map:
            print('Warning %s not in entity vocab map' % e2_str)
        elif ep_str not in entity_pair_map:
            print('Warning %s not in ep vocab map' % ep_str)
        else:
            e1 = entity_map[e1_str]
            e2 = entity_map[e2_str]
            ep = entity_pair_map[ep_str]
            rel = relation_map[rel_str] if rel_str in relation_map else -1
            tokens = [token_map[t] if t in token_map else token_map['<UNK>'] for t in tokens_str]
            if FLAGS.padding > 0 and len(tokens) < FLAGS.max_len:
                padding = [token_map['<PAD>']] * (FLAGS.max_len - len(tokens))
                tokens = tokens + padding if FLAGS.padding == 1 else padding + tokens

            e1_dists = [
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[((i - e1_start) + FLAGS.max_len)]))
                if i < e1_start else
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[FLAGS.max_len]))
                if i < e1_end else
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[((i - e1_end + 1) + FLAGS.max_len)]))
                for i, t in enumerate(tokens)
            ]
            e2_dists = [
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[((i - e2_start) + FLAGS.max_len)]))
                if i < e2_start else
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[FLAGS.max_len]))
                if i < e2_end else
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[((i - e2_end + 1) + FLAGS.max_len)]))
                for i, t in enumerate(tokens)
            ]

            tokens = [tf.train.Feature(int64_list=tf.train.Int64List(value=[t])) for t in tokens]

            example = sequence_example(
                context=features({
                    'e1': int64_feature([e1]),
                    'e2': int64_feature([e2]),
                    'ep': int64_feature([ep]),
                    'rel': int64_feature([rel]),
                    'seq_len': int64_feature([len(tokens)]),
                }),
                feature_lists=feature_lists({
                    "tokens": feature_list(tokens),
                    "e1_dist": feature_list(e1_dists),
                    "e2_dist": feature_list(e2_dists),
                }))

            writer.write(example.SerializeToString())
            return 1
    return 0


def process_file(entity_map, ep_map, rel_map, token_map, total_lines, in_out):
    """
    A function which converts each line of the input file into indices according to the vocabulary
    :param entity_map:
    :param ep_map:
    :param rel_map:
    :param token_map:
    :param total_lines:
    :param in_out:
    :return:
    """
    try:
        in_f, out_path = in_out
        writer = tf.python_io.TFRecordWriter(out_path)
        lines_written = 0
        print('Converting %s to %s' % (in_f, out_path))
        f_reader = gzip.open(in_f, 'rb') if in_f.endswith('.gz') else open(in_f, 'r')
        for i, line in enumerate(f_reader):
            if i % 2500 == 0:
                if not queue.empty():
                    lock.acquire()
                    processed_lines = queue.get(True, .25) + 2500
                    queue.put(processed_lines, True, .25)
                    lock.release()
                    if total_lines > 0:
                        percent_done = 100 * processed_lines / float(total_lines)
                        sys.stdout.write('\rProcessing line %d of %d : %2.2f %%'
                                         % (processed_lines, total_lines, percent_done))
                    else:
                        sys.stdout.write('\rProcessing line %d' % processed_lines)
                    sys.stdout.flush()
            lines_written += make_example(entity_map, ep_map, rel_map, token_map, line, writer)
        f_reader.close()
        writer.close()
        print('\nDone processing %s. Wrote %d lines' % (in_f, lines_written))
    except KeyboardInterrupt:
        return 'KeyboardException'


def get_data_summary(in_files, kg_in_files):
    """
    Given a list of input files get vocabulary from them
    :param in_files:
    :return:
    """
    total_lines = 0
    entity_counter = defaultdict(int)
    ep_counter = defaultdict(int)
    token_counter = defaultdict(int)
    rel_map = {}
    for in_f in in_files:
        if os.path.isfile(in_f):
            line_num = 0
            errors = 0
            print('Updating vocabs for %s' % in_f)
            f_reader = gzip.open(in_f, 'rb') if in_f.endswith('.gz') else open(in_f, 'r')
            update_rel_map = (in_f in kg_in_files)
            for line in f_reader:
                line_num += 1
                if line_num % 1000 == 0:
                    sys.stdout.write('\rProcessing line: %d \t errors: %d ' % (line_num, errors))
                    sys.stdout.flush()
                errors += update_vocab_counts(line, entity_counter, ep_counter, rel_map, token_counter,
                                              update_rel_map)
            print(' Done')
            f_reader.close()
            total_lines += line_num
    return entity_counter, ep_counter, token_counter, rel_map, total_lines


def tsv_to_examples():
    """
    This function is the starting point:
    1). Load the vocabulary if it exists or create a new one
    2). Process the input files using the vocabulary

    :return:
    """
    # Create the output director if it does not exists
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)
    kg_in_files = sorted(glob.glob(FLAGS.kg_in_files))
    text_in_files = sorted(glob.glob(FLAGS.text_in_files))

    # Create the output proto files from the input text files
    in_files = kg_in_files + text_in_files
    out_files = ['%s/%s.proto' % (FLAGS.out_dir, in_f.split('/')[-1]) for in_f in in_files]
    total_lines = 0
    if FLAGS.load_vocab:
        # If the vocabulary is already created, load it
        print('Loading vocab from %s' % FLAGS.load_vocab)
        with open('%s/entities.txt' % FLAGS.load_vocab) as f:
            entity_map = {l.split('\t')[0]: int(l.split('\t')[1]) for l in f}
        with open('%s/ep.txt' % FLAGS.load_vocab) as f:
            ep_map = {l.split('\t')[0]: int(l.split('\t')[1]) for l in f}
        with open('%s/rel.txt' % FLAGS.load_vocab) as f:
            rel_map = {l.split('\t')[0]: int(l.split('\t')[1]) for l in f}
        with open('%s/token.txt' % FLAGS.load_vocab) as f:
            token_map = {l.split('\t')[0]: int(l.split('\t')[1]) for l in f}
        print('Loaded %d tokens, %d entities %d entity-pairs %d relations'
              % (len(token_map), len(entity_map), len(ep_map), len(rel_map)))
    else:
        # Iterate over the data to get the vocabulary
        entity_counter, ep_counter, token_counter, rel_map, total_lines = get_data_summary(in_files, kg_in_files)
        # remove tokens with < min_count
        print('Sorting and filtering vocab maps')
        keep_tokens = sorted([(t, c) for t, c in token_counter.iteritems()
                              if c >= FLAGS.min_count], key=lambda tup: tup[1], reverse=True)
        keep_tokens = [t[0] for t in keep_tokens]
        # int map all the kept vocab strings
        token_map = {t: i for i, t in enumerate(['<PAD>', '<UNK>'] + keep_tokens)}
        entity_map = {e: i for i, e in enumerate(entity_counter.keys())}
        ep_map = {e: i for i, e in enumerate(ep_counter.keys())}

        # export the string->int maps to file
        for f_str, id_map in [('entities', entity_map), ('ep', ep_map), ('rel', rel_map), ('token', token_map)]:
            print('Exporting vocab maps to %s/%s' % (FLAGS.out_dir, f_str))
            with open('%s/%s.txt' % (FLAGS.out_dir, f_str), 'w') as f:
                sorted_id_map = sorted(id_map.items(), key=operator.itemgetter(1))
                [f.write(s + '\t' + str(i) + '\n') for (s, i) in sorted_id_map]

    print('Starting file process threads using %d threads' % FLAGS.num_threads)
    pool = multiprocessing.Pool(FLAGS.num_threads)
    try:
        pool.map_async(partial(process_file, entity_map, ep_map, rel_map, token_map, total_lines),
                       zip(in_files, out_files)).get(999999)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()


def main(argv):
    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    if FLAGS.out_dir == '':
        print('Must supply out_dir')
        sys.exit(1)
    tsv_to_examples()


if __name__ == '__main__':
    tf.app.run()
