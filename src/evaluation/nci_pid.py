from src.evaluation.fb15k import *


class NCIEvaluator(FB15KEvaluator):
    def __init__(self, model, fb15k_dir, kb_str_id_map, kb_id_str_map, entity_str_id_map, entity_id_str_map,
                 ep_str_id_map, ep_id_str_map, session, test_key='valid'):
        super(NCIEvaluator, self).__init__(model, fb15k_dir, kb_str_id_map, kb_id_str_map, entity_str_id_map,
                                           entity_id_str_map, ep_str_id_map, ep_id_str_map, session, test_key)
        self.eval_type = 'NCI'

    def get_scores(self, e1_batch, e2_batch, rel_batch):
        scores = self.session.run([self.model.pos_predictions],
                                  feed_dict={self.model.e1_batch: e1_batch,
                                             self.model.e2_batch: e2_batch,
                                             self.model.kb_batch: rel_batch,
                                             self.model.text_update: False})[0]
        return scores.tolist()

    def eval(self, block=False, take=3000):
        '''
        combine head and tail scores into single ranked list
        '''
        take = min(len(self.splits[self.test_key]), take)
        pos_triples = random.sample(self.splits[self.test_key], take) \
            if take > 0 else self.splits[self.test_key]
        count = float(len(pos_triples))
        average_precision = 0.
        total_hits_at_10 = 0.
        for pos_e1, pos_e2, pos_rel in pos_triples:
            e1_batch = [pos_e1] * len(self.entity_batch)
            e2_batch = [pos_e2] * len(self.entity_batch)
            rel_batch = [pos_rel] * len(self.entity_batch)
            head_scores = self.get_scores(self.entity_batch, e2_batch, rel_batch)
            tail_scores = self.get_scores(e1_batch, self.entity_batch, rel_batch)
            labeled_head_score = [(1, e1, e2, rel, s) if e1 == pos_e1 else (0, e1, e2, rel, s)
                                  for e1, e2, rel, s in zip(self.entity_batch, e2_batch, rel_batch, head_scores)]
            labeled_tail_score = [(1, e1, e2, rel, s) if e2 == pos_e2 else (0, e1, e2, rel, s)
                                  for e1, e2, rel, s in zip(e1_batch, self.entity_batch, rel_batch, tail_scores)]
            all_scores_sorted = sorted(labeled_head_score + labeled_tail_score, key=lambda x: x[4], reverse=True)

            correct_t = 0
            rank = 1.0
            i = 0
            ap = 0
            while correct_t < 2:
                l, e1, e2, rel, s = all_scores_sorted[i]
                fact = (e1, e2, rel)
                if l == 1:
                    correct_t += 1
                    ap += correct_t / rank
                    if rank < 10:
                        total_hits_at_10 += 0.5
                    rank += 1.0
                elif fact not in self.facts:
                    rank += 1.0
                i += 1
            average_precision += (ap/2.0)

        MAP = 100 * (average_precision / count)
        hits_at_10 = 100 * (total_hits_at_10 / count)
        return MAP, hits_at_10

    # def eval(self, block=False, take=3000):
    #     pos_triples = self.splits[self.test_key]
    #     count = len(pos_triples)
    #     all_head_scores = []
    #     all_tail_scores = []
    #     for pos_e1, pos_e2, pos_rel in pos_triples:
    #         e1_batch = [pos_e1] * len(self.entity_batch)
    #         e2_batch = [pos_e2] * len(self.entity_batch)
    #         rel_batch = [pos_rel] * len(self.entity_batch)
    #         head_scores = self.get_scores(self.entity_batch, e2_batch, rel_batch)
    #         tail_scores = self.get_scores(e1_batch, self.entity_batch, rel_batch)
    #         labeled_head_score = [(1, e1, e2, rel, s) if e1 == pos_e1 else (0, e1, e2, rel, s)
    #                               for e1, e2, rel, s in zip(self.entity_batch, e2_batch, rel_batch, head_scores)]
    #         all_head_scores.append(labeled_head_score)
    #         labeled_tail_score = [(1, e1, e2, rel, s) if e2 == pos_e2 else (0, e1, e2, rel, s)
    #                               for e1, e2, rel, s in zip(e1_batch, self.entity_batch, rel_batch, tail_scores)]
    #         all_tail_scores.append(labeled_tail_score)
    #
    #     flat_head_scores = [item for sublist in all_head_scores for item in sublist]
    #     flat_tail_scores = [item for sublist in all_tail_scores for item in sublist]
    #
    #     head_scores_sorted = sorted(flat_head_scores, key=lambda x: x[4], reverse=True)
    #     tail_scores_sorted = sorted(flat_tail_scores, key=lambda x: x[4], reverse=True)
    #
    #     correct_t, i, rank = 0, 0, 1.0
    #     head_ap = 0.0
    #     hits_at_10 = 0
    #     while correct_t < count:
    #         l, e1, e2, rel, s = head_scores_sorted[i]
    #         fact = (e1, e2, rel)
    #         if l == 1:
    #             correct_t += 1
    #             head_ap += correct_t / rank
    #             if rank < 10:
    #                 hits_at_10 += 1
    #             rank += 1.0
    #         elif fact not in self.facts:
    #             rank += 1.0
    #         i += 1
    #
    #     correct_t, i, rank = 0, 0, 1.0
    #     tail_ap = 0.0
    #     while correct_t < count:
    #         l, e1, e2, rel, s = tail_scores_sorted[i]
    #         fact = (e1, e2, rel)
    #         if l == 1:
    #             correct_t += 1
    #             tail_ap += correct_t / rank
    #             if rank < 10:
    #                 hits_at_10 += 1
    #             rank += 1.0
    #         elif fact not in self.facts:
    #             rank += 1.0
    #         i += 1
    #
    #     MAP = 100 * ((head_ap + tail_ap) / (count*2.0))
    #     hits_at_10 = 100 * (hits_at_10/20.0)
    #
    #     return MAP, hits_at_10


    # def eval(self, block=False, take=3000):
    #     '''
    #     Original eval with seperate head and tail ranked lists
    #     '''
    #     take = min(len(self.splits[self.test_key]), take)
    #     pos_triples = random.sample(self.splits[self.test_key], take) if take > 0 else self.splits[self.test_key]
    #     count = float(len(pos_triples) * 2)
    #     total_rank = 0.
    #     total_hits_at_10 = 0.
    #     for pos_e1, pos_e2, pos_rel in pos_triples:
    #         e1_batch = [pos_e1] * len(self.entity_batch)
    #         e2_batch = [pos_e2] * len(self.entity_batch)
    #         rel_batch = [pos_rel] * len(self.entity_batch)
    #         head_rank = self.rank(True, pos_e1, self.entity_batch, e2_batch, rel_batch)
    #         tail_rank = self.rank(False, pos_e2, e1_batch, self.entity_batch, rel_batch)
    #         total_rank += (1 / head_rank)
    #         total_rank += (1 / tail_rank)
    #         if head_rank <= 10:
    #             total_hits_at_10 += 1
    #         if tail_rank <= 10:
    #             total_hits_at_10 += 1
    #     mrr = 100 * (total_rank / count)
    #     hits_at_10 = 100 * (total_hits_at_10 / count)
    #     return mrr, hits_at_10
    #
    # def rank(self, replace_head, pos_idx, e1_batch, e2_batch, rel_batch):
    #     scores = self.session.run([self.model.pos_predictions],
    #                               feed_dict={self.model.e1_batch: e1_batch,
    #                                          self.model.e2_batch: e2_batch,
    #                                          self.model.kb_batch: rel_batch,
    #                                          self.model.text_update: False})[0]
    #     # get rank of the positive triple
    #     n = len(rel_batch)
    #     i = 0
    #     rank = 1
    #     ranked_preds = np.squeeze(scores).argsort()[::-1][:n]
    #
    #     while ranked_preds[i] != pos_idx and i < n:
    #         if replace_head:
    #             e1 = ranked_preds[i]
    #             e2 = e2_batch[0]
    #         else:
    #             e1 = e1_batch[0]
    #             e2 = ranked_preds[i]
    #         fact = (e1, e2, rel_batch[0])
    #         if fact not in self.facts:
    #             rank += 1
    #         i += 1
    #     return float(rank)

class NCIEpEvaluator(FB15KEpEvaluator):
    def __init__(self, model, fb15k_dir, kb_str_id_map, kb_id_str_map, entity_str_id_map, entity_id_str_map,
                 ep_str_id_map, ep_id_str_map, session, test_key='valid'):
        super(NCIEpEvaluator, self).__init__(model, fb15k_dir, kb_str_id_map, kb_id_str_map, entity_str_id_map,
                                             entity_id_str_map, ep_str_id_map, ep_id_str_map, session, test_key)
        self.eval_type = 'NCI'

    def eval(self, block=False, take=1000):
        pos_count = 0.
        neg_count = 0
        all_tail_scores = []
        for pos_e1, pos_e2, pos_rel in self.splits[self.test_key]:
            if (pos_e1, pos_e2) in self.e1_e2_ep_map:
                pos_ep = self.e1_e2_ep_map[(pos_e1, pos_e2)]
                if pos_ep in self.ep_batch_map:
                    pos_count += 1
                    neg_eps = self.ep_batch_map[pos_ep]
                    neg_count += len(neg_eps)
                    ep_batch = [pos_ep] + neg_eps
                    rel_batch = [pos_rel] * (len(ep_batch))
                    tail_scores = self.get_scores(ep_batch, rel_batch)
                    labeled_tail_score = [(1, ep, pos_rel, s) if i == 0 else (0, ep, pos_rel, s)
                                          for i, (ep, s) in enumerate(zip(ep_batch, tail_scores))]
                    all_tail_scores.append(labeled_tail_score)

        print('\n pos_eps: %d   neg eps : %d' % (pos_count, neg_count))

        flat_tail_scores = [item for sublist in all_tail_scores for item in sublist]
        tail_scores_sorted = sorted(flat_tail_scores, key=lambda x: x[3], reverse=True)

        correct_t, i, rank = 0, 0, 1.0
        tail_ap = 0.0
        hits_at_10 = 0
        while correct_t < pos_count:
            l, ep, rel, s = tail_scores_sorted[i]
            e1, e2 = self.ep_e1_e2_map[ep]
            fact = (e1, e2, rel)
            if l == 1:
                correct_t += 1
                tail_ap += correct_t / rank
                if rank < 10:
                    hits_at_10 += 1
                rank += 1.0
            elif fact not in self.facts:
                rank += 1.0
            i += 1

        MAP = 100 * (tail_ap / pos_count)
        hits_at_10 = 100 * (hits_at_10/10.0)
        return MAP, hits_at_10

    def get_scores(self, ep_batch, rel_batch):
        feed_dict = {self.model.ep_batch: ep_batch, self.model.kb_batch: rel_batch, self.model.text_update: False}
        if self.model.model_type == 'joint_entity_pair_model':
            e1_batch, e2_batch = zip(*[self.ep_e1_e2_map[ep] for ep in ep_batch])
            feed_dict[self.model.e1_batch] = e1_batch
            feed_dict[self.model.e2_batch] = e2_batch
        scores = self.session.run([self.model.pos_predictions], feed_dict=feed_dict)[0]
        return scores.tolist()




