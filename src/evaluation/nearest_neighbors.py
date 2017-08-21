from __future__ import print_function
import numpy as np
import sys


class NearestNeighbors(object):
    def __init__(self, model, session, kb_str_id_map, kb_id_str_map,
                 ep_str_id_map, ep_id_str_map, entity_str_id_map, entity_id_str_map):
        self.eval_type = 'nearest_neighbor'
        self.model = model
        self.session = session
        self.kb_str_id_map = kb_str_id_map
        self.kb_id_str_map = kb_id_str_map
        self.ep_str_id_map = ep_str_id_map
        self.ep_id_str_map = ep_id_str_map
        self.entity_str_id_map = entity_str_id_map
        self.entity_id_str_map = entity_id_str_map

    def matrix_similarity(self, nn_op, feed_dict, id_str_map, k):
        sim_matrix = self.session.run([nn_op], feed_dict=feed_dict)[0]
        batch_vals = feed_dict.values()[0]
        for i in range(sim_matrix.shape[0]):
            print(id_str_map[batch_vals[i]])
            scores = sim_matrix[i, :]
            top_k_ids = scores.argsort()[-k:][::-1]
            top_k_strs = [id_str_map[_id] for j, _id in enumerate(top_k_ids) if j > 0]
            print(top_k_strs)
            # print(top_k_ids)
        # print(len(id_str_map))

    def kb_similarity(self, k=5, max_queries=50):
        rand_relations = np.random.randint(self.model._kb_size, size=max_queries, )
        feed_dict = {self.model.kb_batch: rand_relations}
        self.matrix_similarity(self.model.kb_nearest_neighbor, feed_dict, self.kb_id_str_map, k)

    def entity_similarity(self, k=5, max_queries=50):
        if self.model.model_type == 'entity_pair_model' or self.model.model_type == 'classifier':
            id_str_map = self.ep_id_str_map
            batch_key = self.model.ep_batch
        elif self.model.model_type == 'entity_model':
            id_str_map = self.entity_id_str_map
            batch_key = self.model.e1_batch
        else:
            print('Only supports entity and entity pair models')
            sys.exit(1)
        rand_entities = np.random.randint(len(id_str_map), size=max_queries)
        feed_dict = {batch_key: rand_entities}
        self.matrix_similarity(self.model.entity_nearest_neighbor, feed_dict, id_str_map, k)