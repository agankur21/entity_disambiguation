import sys
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict
from random import shuffle

FLAGS = tf.app.flags.FLAGS


class Batcher(object):
    def __init__(self, in_file, num_epochs, max_seq, batch_size):
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._max_seq = max_seq
        self._step = 1.
        self.in_file = in_file
        self.next_batch_op = self.input_pipeline(in_file, self._batch_size, num_epochs=num_epochs)

    def next_batch(self, sess):
        return sess.run(self.next_batch_op)

    def input_pipeline(self, file_pattern, batch_size, num_epochs=None):
        filenames = tf.matching_files(file_pattern)
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        parsed_batch = self.example_parser(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 12 * batch_size
        next_batch = tf.train.batch(
            parsed_batch, batch_size=batch_size, capacity=capacity,
            num_threads=10, dynamic_pad=True, allow_smaller_final_batch=True)
        return next_batch

    def example_parser(self, filename_queue):
        return []


class GraphBatcher(Batcher):
    def __init__(self, in_file, num_epochs, max_seq, batch_size):
        super(GraphBatcher, self).__init__(in_file, num_epochs, max_seq, batch_size)

    def example_parser(self, filename_queue):
        reader = tf.TFRecordReader()
        key, record_string = reader.read(filename_queue)

        # Define how to parse the example
        context_features = {
            'e1': tf.FixedLenFeature([], tf.int64),
            'e2': tf.FixedLenFeature([], tf.int64),
            'ep': tf.FixedLenFeature([], tf.int64),
            'rel': tf.FixedLenFeature([], tf.int64),
            'seq_len': tf.FixedLenFeature([], tf.int64),
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "e1_dist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "e2_dist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=record_string,
                                                                           context_features=context_features,
                                                                           sequence_features=sequence_features)
        e1 = context_parsed['e1']
        e2 = context_parsed['e2']
        ep = context_parsed['ep']
        rel = context_parsed['rel']
        tokens = sequence_parsed['tokens']
        e1_dist = sequence_parsed['e1_dist']
        e2_dist = sequence_parsed['e2_dist']
        seq_len = context_parsed['seq_len']

        return [e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len]


class MentionBatcher(Batcher):
    def __init__(self, in_file, num_epochs, max_seq, batch_size):
        super(MentionBatcher, self).__init__(in_file, num_epochs, max_seq, batch_size)

    def example_parser(self, filename_queue):
        reader = tf.TFRecordReader()
        key, record_string = reader.read(filename_queue)
        # Define how to parse the example
        context_features = {
            'seq_len_left': tf.FixedLenFeature([], tf.int64),
            'seq_len_right': tf.FixedLenFeature([], tf.int64),
        }
        sequence_features = {
            "left_tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "right_tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "mentions": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=record_string,
                                                                           context_features=context_features,
                                                                           sequence_features=sequence_features)
        seq_len_left = context_parsed['seq_len_left']
        seq_len_right = context_parsed['seq_len_left']
        left_tokens = sequence_parsed['left_tokens']
        right_tokens = sequence_parsed['right_tokens']
        mentions = sequence_parsed['mentions']
        labels = sequence_parsed['labels']
        return [seq_len_left, seq_len_right, left_tokens, right_tokens, mentions, labels]


class InMemoryGraphBatcher(GraphBatcher):
    def __init__(self, in_file, num_epochs, max_seq, batch_size):
        super(InMemoryGraphBatcher, self).__init__(in_file, num_epochs, max_seq, batch_size)
        self.epoch = 0.
        loading_batch_size = 1000
        self.next_batch_op = self.input_pipeline(in_file, loading_batch_size, num_epochs=1)
        self.data = defaultdict(list)
        self._starts = {}
        self._ends = {}
        self._bucket_probs = {}

    def load_all_data(self, sess, max_batches=-1):
        '''
        load batches to memory for shuffling and dynamic padding
        '''
        batch_num = 0
        samples = 0
        start_time = time.time()
        print ('Loading data from %s' % self.in_file)
        try:
            while max_batches <= 0 or batch_num < max_batches:
                batch = sess.run(self.next_batch_op)
                e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len = batch
                samples += e1.shape[0]
                self.data[tokens.shape[1]].append(batch)
                batch_num += 1
                sys.stdout.write('\rLoading batch: %d' % batch_num)
                sys.stdout.flush()
        except:
            print('')
        for seq_len, batches in self.data.iteritems():
            self.data[seq_len] = [tuple((e1[i], e2[i], ep[i], rel[i], tokens[i], e1d[i], e2d[i], sl[i]))
                                  for (e1, e2, ep, rel, tokens, e1d, e2d, sl) in batches
                                  for i in range(e1.shape[0])]
        self.reset_batch_pointer()
        end_time = time.time()
        print('Done, loaded %d samples in %5.2f seconds' % (samples, (end_time - start_time)))
        return batch_num

    def next_batch(self, sess):
        # select bucket to create batch from
        self.step += 1
        bucket = self.select_bucket()
        batch = self.data[bucket][self._starts[bucket]:self._ends[bucket]]
        # update pointers
        self._starts[bucket] = self._ends[bucket]
        self._ends[bucket] = min(self._ends[bucket] + self._batch_size, len(self.data[bucket]))
        self._bucket_probs[bucket] = max(0, len(self.data[bucket]) - self._starts[bucket])

        # TODO this is dumb
        _e1 = np.array([e1 for e1, e2, ep, rel, t, e1d, e2d, s in batch])
        _e2 = np.array([e2 for e1, e2, ep, rel, t, e1d, e2d, s in batch])
        _ep = np.array([ep for e1, e2, ep, rel, t, e1d, e2d, s in batch])
        _rel = np.array([rel for e1, e2, ep, rel, t, e1d, e2d, s in batch])
        _tokens = np.array([t for e1, e2, ep, rel, t, e1d, e2d, s in batch])
        _e1d = np.array([e1d for e1, e2, ep, rel, t, e1d, e2d, s in batch])
        _e2d = np.array([e2d for e1, e2, ep, rel, t, e1d, e2d, s in batch])
        _seq_len = np.array([s for e1, e2, ep, rel, t, e1d, e2d, s in batch])
        batch = (_e1, _e2, _ep, _rel, _tokens, _e1d, _e2d, _seq_len)
        if sum(self._bucket_probs.itervalues()) == 0:
            self.reset_batch_pointer()
        return batch

    def reset_batch_pointer(self):
        # shuffle each bucket
        for bucket in self.data.itervalues():
            shuffle(bucket)
        self.epoch += 1
        self.step = 0.
        # print('\nStarting epoch %d' % self.epoch)
        self._starts = {i: 0 for i in self.data.iterkeys()}
        self._ends = {i: min(self._batch_size, len(examples)) for i, examples in self.data.iteritems()}
        self._bucket_probs = {i: len(l) for (i, l) in self.data.iteritems()}

    def select_bucket(self):
        buckets, weights = zip(*[(i, p) for i, p in self._bucket_probs.iteritems() if p > 0])
        total = float(sum(weights))
        probs = [w / total for w in weights]
        bucket = np.random.choice(buckets, p=probs)
        return bucket


def rowless_example_parser(filename_queue):
    reader = tf.TFRecordReader()
    key, record_string = reader.read(filename_queue)

    # Define how to parse the example
    context_features = {
        'e1': tf.FixedLenFeature([], tf.int64),
        'e2': tf.FixedLenFeature([], tf.int64),
        'ep': tf.FixedLenFeature([], tf.int64),
        'rel': tf.FixedLenFeature([], tf.int64),
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "e1_dist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "e2_dist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        'seq_len': tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=record_string,
                                                                       context_features=context_features,
                                                                       sequence_features=sequence_features)
    e1 = context_parsed['e1']
    e2 = context_parsed['e2']
    ep = context_parsed['ep']
    rel = context_parsed['rel']
    tokens = sequence_parsed['tokens']
    e1_dist = sequence_parsed['e1_dist']
    e2_dist = sequence_parsed['e2_dist']
    seq_len = sequence_parsed['seq_len']

    return [e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len]


class RowlessGraphBatcher(GraphBatcher):
    def example_parser(self, filename_queue):
        return rowless_example_parser(filename_queue)


class RowlessInMemoryBatcher(InMemoryGraphBatcher):
    def example_parser(self, filename_queue):
        return rowless_example_parser(filename_queue)


class RowlessEPBatcher(RowlessGraphBatcher):
    def __init__(self, in_file, num_epochs, max_seq, batch_size, ep_filter=None):
        super(RowlessEPBatcher, self).__init__(in_file, num_epochs, max_seq, batch_size)
        self.epoch = 0.
        self.next_batch_op = self.input_pipeline(in_file, batch_size, num_epochs=1)
        self.ep_data = {}
        self.ep_filter = ep_filter

    def load_all_data(self, sess, max_batches=-1):
        '''
        load batches to memory for shuffling and dynamic padding
        '''
        batch_num = 0
        samples = 0
        start_time = time.time()
        print ('Loading data from %s' % self.in_file)
        batches = []
        try:
            while max_batches <= 0 or batch_num < max_batches:
                batch = sess.run(self.next_batch_op)
                e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len = batch
                batches.append(zip(e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len))
                batch_num += 1
                sys.stdout.write('\rLoading batch: %d' % batch_num)
                sys.stdout.flush()
                samples += batch[0].shape[0]
        except:
            print('')
        flat_batches = [item for sublist in batches for item in sublist]
        e1, e2, ep, rel, tokens, e1_dist, e2_dist, seq_len = zip(*flat_batches)
        self.ep_data = {_ep: (e1[i], e2[i], _ep, rel[i], tokens[i], e1_dist[i], e2_dist[i], seq_len[i])
                        for i, _ep in enumerate(ep) if (self.ep_filter is None or int(_ep) in self.ep_filter)}

        end_time = time.time()
        ep_counts = np.array([len(examples) for ep, examples in self.ep_data.iteritems()])
        print('Done, loaded %d samples from %d entity pairs in %5.2f seconds'
              % (samples, len(self.ep_data), (end_time - start_time)))
        if len(self.ep_data) > 0:
            print(
                'Ep count stats: min %d  mean %d  max %d' % (np.min(ep_counts), np.mean(ep_counts), np.max(ep_counts)))
        return batch_num
