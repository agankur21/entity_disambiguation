import numpy as np
import random


class FB15KEvaluator(object):
    def __init__(self, model, fb15k_dir, kb_str_id_map, kb_id_str_map,
                 entity_str_id_map, entity_id_str_map, ep_str_id_map, ep_id_str_map,
                 session, test_key='valid'):
        self.eval_type = 'fb15K-237'
        self.model = model
        self.session = session
        self.kb_str_id_map = kb_str_id_map
        self.kb_id_str_map = kb_id_str_map
        self.entity_str_id_map = entity_str_id_map
        self.entity_id_str_map = entity_id_str_map
        self.ep_str_id_map = ep_str_id_map
        self.ep_id_str_map = ep_id_str_map
        self.test_key = test_key
        self.facts = set()
        self.splits = {}
        # read in data
        for f_name in ['train', 'valid', 'test']:
            with open(fb15k_dir + '/' + f_name + '.txt', 'r') as f:
                # randomly sample k lines from the file
                lines = [l.strip().split('\t')[:3] for l in f.readlines()]
                filtered_lines = [(e1, e2, rel) for (e1, e2, rel) in lines if
                                  e1 in self.entity_str_id_map and e2 in self.entity_str_id_map]
                print('filtered : ' + str(len(lines) - len(filtered_lines)))
                self.splits[f_name] = [(self.entity_str_id_map[e1], self.entity_str_id_map[e2], self.kb_str_id_map[rel])
                                       for e1, e2, rel in filtered_lines]
                self.facts = self.facts.union(self.splits[f_name])
        self.entity_batch = [i for i in range(len(entity_str_id_map))]

    def eval(self, block=False, take=3000):
        take = min(len(self.splits[self.test_key]), take)
        pos_triples = random.sample(self.splits[self.test_key], take) if take > 0 else self.splits[self.test_key]
        count = float(len(pos_triples))
        total_rank = 0.
        total_hits_at_10 = 0.
        for pos_e1, pos_e2, pos_rel in pos_triples:
            e1_batch = [pos_e1] * len(self.entity_batch)
            rel_batch = [pos_rel] * len(self.entity_batch)
            tail_rank = self.rank(False, pos_e2, e1_batch, self.entity_batch, rel_batch)
            total_rank += (1 / tail_rank)
            if tail_rank <= 10:
                total_hits_at_10 += 1
        print('--- positive: %d   negative: %d' % (take, len(self.entity_batch)))

        mrr = 100 * (total_rank / count)
        hits_at_10 = 100 * (total_hits_at_10 / count)
        return mrr, hits_at_10

    def rank(self, replace_head, pos_idx, e1_batch, e2_batch, rel_batch):
        scores = self.session.run([self.model.pos_predictions],
                                  feed_dict={self.model.e1_batch: e1_batch,
                                             self.model.e2_batch: e2_batch,
                                             self.model.kb_batch: rel_batch,
                                             self.model.text_update: False})[0]
        # get rank of the positive triple
        n = len(rel_batch)
        i = 0
        rank = 1
        ranked_preds = np.squeeze(scores).argsort()[::-1][:n]

        while ranked_preds[i] != pos_idx and i < n:
            if replace_head:
                e1 = ranked_preds[i]
                e2 = e2_batch[0]
            else:
                e1 = e1_batch[0]
                e2 = ranked_preds[i]
            fact = (e1, e2, rel_batch[0])
            if fact not in self.facts:
                rank += 1
            i += 1
        return float(rank)


