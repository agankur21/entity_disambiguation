from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import operator
import sys
import os
from collections import defaultdict

import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../.."))
from src.preprocessing.parser import *

########################Defining input flags #########################################
tf.app.flags.DEFINE_string('text_in_files', '', 'input text file in pubtator format')
tf.app.flags.DEFINE_string('out_dir', '', 'export tf protos')
tf.app.flags.DEFINE_string('load_vocab', '', 'directory containing vocab files to load')
tf.app.flags.DEFINE_integer('max_len', 20, 'maximum sequence length')
tf.app.flags.DEFINE_integer('min_count', 5, 'replace tokens occuring less than this many times with <UNK>')
tf.app.flags.DEFINE_integer('num_threads', 5, 'max number of threads to use for parallel processing')
tf.app.flags.DEFINE_boolean('padding', 0, '0: no padding, 1: 0 pad to the right of seq, 2: 0 pad to the left')
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


#################################################################################################

def update_text_vocab(line, token_counter, mention_map):
    """
    Parse each line containing the mention
    :param line:
    :param token_counter:
    :param mention_map:
    :return:
    """
    # Remove double spaces in the line
    line = remove_double_spaces(line)
    tokens = line.strip().split()
    flag_in_mention = False
    mention_list = []
    for token in tokens:
        if token == MENTION_END:
            if len(mention_list) > 0:
                mention = ' '.join(mention_list)
                mention_list = []
                if mention not in mention_map:
                    mention_map[mention] = len(mention_map)
            flag_in_mention = False
        elif token == MENTION_START:
            flag_in_mention = True
        elif flag_in_mention:
            mention_list.append(token)
        if token != MENTION_START and token != MENTION_END:
            token_counter[token] += 1


def update_entity_vocab(entity_list, entity_map):
    """
    Update the entity map from the file
    :param line:
    :param entity_map:
    :return:
    """
    for entity in entity_list:
        for sdui in entity.sdui_set:
            if sdui not in entity_map:
                entity_map[sdui] = len(entity_map)


def get_vocab_from_data(pubtator_data):
    """
    Given data in pubtator format and extract vocabulary from it and write it into a directory
    :param pubtator_data:
    :param out_dir
    :return:
    """
    entity_map = {PAD_STR: 0, OOV_STR: 1}
    mention_map = {PAD_STR: 0, OOV_STR: 1}
    token_counter = defaultdict(int)
    update_entity_vocab(pubtator_data.entity_list, entity_map)
    for sentence_list in pubtator_data.pmid_sentence_map.values():
        for sentence in sentence_list:
            update_text_vocab(sentence.text, token_counter, mention_map)
    print('Sorting and filtering vocab maps')
    keep_tokens = sorted([(t, c) for t, c in token_counter.iteritems() if c >= FLAGS.min_count], key=lambda tup: tup[1],
                         reverse=True)
    keep_tokens = [t[0] for t in keep_tokens]
    token_map = {t: i for i, t in enumerate([PAD_STR, OOV_STR, SENT_START, SENT_END] + keep_tokens)}
    # export the string->int maps to file
    for f_str, id_map in [('entities', entity_map), ('mention', mention_map), ('token', token_map)]:
        print('Exporting %s vocab maps to %s/%s' % (f_str, FLAGS.out_dir, f_str))
        with open('%s/%s.txt' % (FLAGS.out_dir, f_str), 'w') as f:
            sorted_id_map = sorted(id_map.items(), key=operator.itemgetter(1))
            [f.write(s + '\t' + str(i) + '\n') for (s, i) in sorted_id_map]
    return entity_map, token_map, mention_map


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


#####################################################################################################################

def truncate(tokens, max_length, direction):
    if direction == 'right':
        return tokens[len(tokens) - max_length:]
    else:
        return tokens[0:max_length]


def encode_text(text, token_map, use_padding=True, padding_type='right'):
    """
    Encode
    :param text:
    :param token_map:
    :param use_padding:
    :param padding_type:
    :return:
    """
    tokens_str = remove_double_spaces(text).split()
    sentence_length = min(FLAGS.max_len, len(tokens_str))
    tokens = [token_map[t] if t in token_map else token_map[OOV_STR] for t in tokens_str]
    if use_padding:
        if len(tokens_str) > FLAGS.max_len:
            tokens = truncate(tokens, FLAGS.max_len, direction=padding_type)
        else:
            padding = [token_map[PAD_STR]] * (FLAGS.max_len - len(tokens))
            if padding_type == 'right':
                tokens = tokens + padding
            else:
                tokens = padding + tokens
    return sentence_length, tokens


def encode_set(item_set, item_map):
    out = set([])
    for item in item_set:
        if item in item_map:
            out.add(item_map[item])
        else:
            out.add(item_map[OOV_STR])
    return out


def make_example(entity_map, token_map, mention_map, list_sentence, writer):
    """
    Steps of converting into examples:
       -- Get the mention vector for all the mentions in the list_sentences
       -- For each sentence :
             Generate 3 Vectors : -Left sentence, right sentence and vector of entities from sdui map
    """
    num_lines_written = 0
    try:
        mention_set = set([x.entity.mention_str for x in list_sentence])
        tf_mentions = [tf.train.Feature(int64_list=tf.train.Int64List(value=[t])) for t in
                       encode_set(mention_set, mention_map)]
        for sentence in list_sentence:
            left_sentence_length, encoded_left_tokens = encode_text(sentence.left_component, token_map,
                                                                    padding_type='right')
            tf_tokens_left = [tf.train.Feature(int64_list=tf.train.Int64List(value=[t])) for t in encoded_left_tokens]
            right_sentence_length, encoded_right_tokens = encode_text(sentence.right_component, token_map,
                                                                      padding_type='left')
            tf_tokens_right = [tf.train.Feature(int64_list=tf.train.Int64List(value=[t])) for t in
                               encoded_right_tokens[::-1]]
            tf_entity_labels = [tf.train.Feature(int64_list=tf.train.Int64List(value=[t])) for t in
                                encode_set(sentence.entity.sdui_set, entity_map)]
            example = sequence_example(
                context=features({
                    'seq_len_left': int64_feature([left_sentence_length]),
                    'seq_len_right': int64_feature([right_sentence_length])
                }),
                feature_lists=feature_lists(
                    dict(left_tokens=feature_list(tf_tokens_left), right_tokens=feature_list(tf_tokens_right),
                         mentions=feature_list(tf_mentions), labels=feature_list(tf_entity_labels))))
            writer.write(example.SerializeToString())
            num_lines_written += 1
    except Exception as e:
        print(e.message)
    return num_lines_written


###################################################################################################################

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
    text_in_files = FLAGS.text_in_files
    if not FLAGS.load_vocab:
        pubtator_data = PubtatorData.parse_pubtator_data(text_in_files, mode='train')
        entity_map, token_map, mention_map = get_vocab_from_data(pubtator_data)
    else:
        entity_map, token_map, mention_map = load_vocab(FLAGS.load_vocab)

    for mode in ['train', 'dev', 'test']:
        pubtator_data = PubtatorData.parse_pubtator_data(text_in_files, mode=mode)
        out_file_path = '%s/%s.proto' % (FLAGS.out_dir, mode)
        writer = tf.python_io.TFRecordWriter(out_file_path)
        try:
            lines_written = 0
            for pmid, list_sentence in pubtator_data.pmid_sentence_map.iteritems():
                lines_written += make_example(entity_map, token_map, mention_map, list_sentence, writer)
                if lines_written % 50 == 0:
                    print("Successfully written %d lines to the file : %s" % (lines_written, out_file_path))
        except Exception as e:
            print(e.message)
        finally:
            writer.close()


def main(argv):
    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].iteritems()])))
    if FLAGS.text_in_files == '':
        print('Must supply input text file directory')
        sys.exit(1)
    if FLAGS.out_dir == '':
        print('Must supply out_dir')
        sys.exit(1)
    tsv_to_examples()


if __name__ == '__main__':
    tf.app.run()
