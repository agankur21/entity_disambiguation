import re
import numpy as np
import subprocess
from sklearn.metrics import average_precision_score

class TACEvaluator(object):
    def __init__(self, model, rel_map, token_map, session, candidate_file, logdir,
                 norm_digits=True, center_only=True, arg_entities=False, year=2012, max_len=20):
        self.eval_type = 'TAC'
        self.model = model
        self.session = session
        self.rel_map = rel_map
        self.token_map = token_map
        self.facts = set()
        self.norm_digits = norm_digits
        self.center_only = center_only
        self.arg_entities = arg_entities
        self.year = year
        self.max_len = max_len

        output_lines, sentences, tac_rels, padded_tokens, padded_e1_dist, padded_e2_dist, seq_lens \
            = self.process_candidate_file(candidate_file)
        self.padded_tokens = np.array(padded_tokens)
        self.e1_dists = np.array(padded_e1_dist)
        self.e2_dists = np.array(padded_e2_dist)
        self.seq_lens = np.array(seq_lens)
        self.output_lines = np.array(output_lines)
        self.sentences = sentences
        self.tac_rels = np.array([self.rel_map[tac_rel] if tac_rel in self.rel_map else 0 for tac_rel in tac_rels])
        self.eval_id = 0
        self.logdir = logdir

    def process_line(self, l, token_map):
        try:
            query_id, tac_rel, sf_2, doc_info, start_1, end_1, start_2, end_2, pattern = l.strip().split('\t')
            tokens = re.sub(r'[0-9]', '0', pattern).split()

            e1_start = int(start_1)
            e1_end = int(end_1)
            e2_start = int(start_2)
            e2_end = int(end_2)
            if e1_start < e2_start:
                s1, e1, s2, e2, arg1_first = e1_start, e1_end, e2_start, e2_end, True
            else:
                s1, e1, s2, e2, arg1_first = e2_start, e2_end, e1_start, e1_end, False

            if self.arg_entities or self.center_only:
                left = tokens[:s1]
                center = tokens[e1:s2]
                right = tokens[e2:]
                first_arg, second_arg = ('$ARG1', '$ARG2') if arg1_first else ('$ARG2', '$ARG1')
                if self.center_only:
                    tokens = [first_arg] + center + [second_arg]
                else:
                    tokens = left + [first_arg] + center + [second_arg] + right

                e1_start = tokens.index('$ARG1')
                e1_end = e1_start + 1
                e2_start = tokens.index('$ARG2')
                e2_end = e2_start + 1

            e1_dists = [((i - e1_start) + self.max_len) if i < e1_start
                        else self.max_len if i < e1_end
            else ((i - e1_end + 1) + self.max_len)
                        for i, t in enumerate(tokens)]

            e2_dists = [((i - e2_start) + self.max_len) if i < e2_start
                        else self.max_len if i < e2_end
            else ((i - e2_end + 1) + self.max_len)
                        for i, t in enumerate(tokens)]

            tokens_mapped = [token_map[t] if t in token_map
                             else token_map['<UNK>']
                             for t in tokens]
            out_line = '\t'.join([query_id, tac_rel, sf_2, doc_info, start_1, end_1, start_2, end_2])

            return out_line, pattern, tac_rel, tokens_mapped, e1_dists, e2_dists, len(tokens_mapped)
        except Exception as e:
            print('error: ' + str(e))
            return None

    def process_candidate_file(self, candidate_file):
        with open(candidate_file, 'r') as f:
            # process each line of the file and filter by max len
            processed_lines = [ll for ll in [self.process_line(l, self.token_map) for l in f] if ll]
            print(len(processed_lines), self.max_len)
            output_lines, sentences, tac_rels, tokens, e1_dists, e2_dists, seq_lens = \
                zip(*[(l, p, r, t, e1, e2, s) for (l, p, r, t, e1, e2, s) in processed_lines if s <= self.max_len])
            # pad sequences up to max len
            pad_token = self.token_map['<PAD>']
            dist_pad = 0
            padded_tokens = [t + [pad_token] * (self.max_len - len(t)) for t in tokens]
            padded_e1_dist = [t + [dist_pad] * (self.max_len - len(t)) for t in e1_dists]
            padded_e2_dist = [t + [dist_pad] * (self.max_len - len(t)) for t in e2_dists]
            return output_lines, sentences, tac_rels, padded_tokens, padded_e1_dist, padded_e2_dist, seq_lens

    def _score_candidate_file(self, scoring_op, max_batch_size):
        max_batch_size = min(max_batch_size, len(self.padded_tokens))
        start = 0
        end = min(max_batch_size, len(self.padded_tokens))
        score_list = []
        while start != end:
            token_batch = self.padded_tokens[start:end]
            e1_dist_batch = self.e1_dists[start:end]
            e2_dist_batch = self.e2_dists[start:end]
            seq_len_batch = self.seq_lens[start:end]
            kb_batch = self.tac_rels[start:end]
            feed_dict = {
                self.model.text_batch: token_batch,
                self.model.kb_batch: kb_batch,
                self.model.e1_dist_batch: e1_dist_batch,
                self.model.e2_dist_batch: e2_dist_batch,
                self.model.seq_len_batch: seq_len_batch,
            }
            batch_scores = self.session.run([scoring_op], feed_dict=feed_dict)[0]
            score_list.append(batch_scores)
            start = end
            end = min(end + max_batch_size, len(self.padded_tokens))
        scores = [item for sublist in score_list for item in sublist]
        return scores

    def score_candidate_file(self, max_batch_size=25000):
        scores = self._score_candidate_file(self.model.text_kb, max_batch_size)
        scored_file = '%s/%d' % (self.logdir, self.eval_id)
        out_scores = sorted(zip(self.output_lines, scores), key=lambda x: x[1])
        with open(scored_file, 'w') as f:
            for line, score in out_scores:
                f.write('%s\t%.5f\n' % (line, score))
        self.eval_id += 1
        return scored_file

    def variance_candidate_file(self, max_batch_size=1000):
        print('--- padded token length: %d' % len(self.padded_tokens))
        scores = self._score_candidate_file(self.model.text_variance, max_batch_size)
        print('--- scores length: %d' % len(scores))
        scored_file = '%s/variance_%d' % (self.logdir, self.eval_id)
        out_scores = sorted(zip(self.output_lines, self.sentences, scores), key=lambda x: x[2])

        label_index = 3
        scores = [score for line, sentence, score in out_scores]
        labels = [int(line.split('\t')[label_index]) for line, sentence, score in out_scores]
        p_at_10 = float(np.mean(labels[:10])*100)
        p_at_100 = float(np.mean(labels[:100])*100)
        p_at_1000 = float(np.mean(labels[:1000])*100)
        avg_p = average_precision_score(labels, 1.0-np.array(scores))*100
        correct = [(s, l) for s, l in zip(scores, labels) if (s >= .5 and l == 0) or (s < .5 and l == 1)]
        accuracy = (float(len(correct)) / float(len(scores))) * 100
        print('accuracy: %2.2f p@10: %2.2f p@100: %2.2f p@1000: %2.2f avg_p: %2.2f'
              % (accuracy, p_at_10, p_at_100, p_at_1000, avg_p))

        with open(scored_file, 'w') as f:
            for line, sentence, score in out_scores:
                f.write('%s\t%s\t%.5f\n' % (line, sentence, score))
        self.eval_id += 1
        return scored_file, avg_p

    def eval(self, block=False):
        print('\nRunning TAC evaluation')
        scored_file = self.score_candidate_file()
        tuned_dir = '%s/tuned_tac/%d' % (self.logdir, self.eval_id)
        print('%s %d %s %s' % ("bin/tac-evaluation/tune-thresh-prescored.sh", self.year, scored_file, tuned_dir))

        process = subprocess.Popen(
                ["bin/tac-evaluation/tune-thresh-prescored.sh", str(self.year), scored_file, tuned_dir],
                stdout=subprocess.PIPE)
        if block:
            out, err = process.communicate()
            f1 = float(out.split(':')[-1]) * 100
            print('F1 : %2.3f' % f1)
            return f1
        else:
            return -1


class RowlessTACEvaluator(TACEvaluator):
    def _score_candidate_file(self, scoring_op, max_batch_size):
        max_batch_size = min(max_batch_size, len(self.padded_tokens))
        start = 0
        end = min(max_batch_size, len(self.padded_tokens))
        score_list = []
        while start != end:
            token_batch = self.padded_tokens[start:end]
            e1_dist_batch = self.e1_dists[start:end]
            e2_dist_batch = self.e2_dists[start:end]
            seq_len_batch = np.expand_dims(self.seq_lens[start:end], 1)
            kb_batch = self.tac_rels[start:end]
            context_batch = range(seq_len_batch.shape[0])
            feed_dict = {
                self.model.text_batch: token_batch,
                self.model.kb_batch: kb_batch,
                self.model.e1_dist_batch: e1_dist_batch,
                self.model.e2_dist_batch: e2_dist_batch,
                self.model.seq_len_batch: seq_len_batch,
                self.model.context_indices: context_batch,
            }
            batch_scores = self.session.run([scoring_op], feed_dict=feed_dict)[0]
            score_list.append(batch_scores)
            start = end
            end = min(end + max_batch_size, len(self.padded_tokens))
        scores = [item for sublist in score_list for item in sublist]
        return scores

