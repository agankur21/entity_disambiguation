import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
from normalizer import *
from data_cleaning import *
from src.models.umls import MRConso


class DataSummarizer:
    """
    This class aims to generate a summary of the datasets
    1).Basic input data summary
    """

    def __init__(self, input_file=None, normalizer=None):
        if input_file is None:
            input_file = os.path.join(TRAINING_DATA_PATH, CORPUS_FILENAME)
        self.normalizer = normalizer
        self.pubtator_data = PubtatorDataEntityMentions.parse_pubtator_data(input_file, self.normalizer)
        self.mrconso_data = MRConso.parse_database(os.path.join(DATA_DIR, "umls", "umls_concepts.txt"), normalizer)
        self.str_to_sdui_mapping = self._get_str_sdui_actual_mapping()

    def _get_str_sdui_actual_mapping(self):
        str_to_sdui_mapping = {}
        for row in self.mrconso_data:
            if row.processed_str not in str_to_sdui_mapping:
                str_to_sdui_mapping[row.processed_str] = set([])
            str_to_sdui_mapping[row.processed_str].add(row.sdui)
        return str_to_sdui_mapping

    @print_exception
    def print_input_data_summary(self):
        print("Total Number of SDUI's : %d" % len(self.pubtator_data.sdui_map))
        print ("Total Number of Mentions :%d" % len(self.pubtator_data.entity_list))
        print ("Total Number of Distinct Mentions :%d" % len(self.pubtator_data.merged_entity_list))
        entities_without_mapping = filter(lambda x: len(x.sdui_set) == 0, self.pubtator_data.entity_list)
        print ("Total number of Mentions with No MESH Mapping :%d" % len(entities_without_mapping))
        entities_with_single_mapping = filter(lambda x: len(x.sdui_set) == 1, self.pubtator_data.entity_list)
        print ("Mentions with single MESH Mapping :%d" % len(entities_with_single_mapping))
        num_distinct_mentions_without_mapping = len(set([x.normalized_mention for x in entities_without_mapping]))
        print ("Total number of Distinct Mentions with No MESH Mapping :%d" % num_distinct_mentions_without_mapping)
        mentions_with_ambiguity = filter(lambda x: x.is_ambiguous, self.pubtator_data.merged_entity_list)
        print ("Total number of Distinct mentions with ambiguity : %d" % len(mentions_with_ambiguity))
        print ("For all entities with at least single MSH id ")
        self.get_mention_match(self.pubtator_data.entity_list)
        print ("For distinct entities with at least single MSH id")
        self.get_mention_match(self.pubtator_data.merged_entity_list)

    @print_exception
    def get_mention_match(self, entity_list):
        num_mention_match = 0
        num_mention_match_sdui_match = 0
        for entity in entity_list:
            if entity.normalized_mention in self.str_to_sdui_mapping and len(entity.sdui_set) > 0:
                num_mention_match += 1
                num_mention_match_sdui_match = num_mention_match_sdui_match + 1 if len(entity.sdui_set.intersection(
                    self.str_to_sdui_mapping[entity.normalized_mention])) > 0 else num_mention_match_sdui_match
        print ("Total number of mentions matching in database : %d" % num_mention_match)
        print ("Total number of mentions and at least one sdui match: %d" % num_mention_match_sdui_match)


if __name__ == '__main__':
    print("Stats without string cleaning")
    normalizer = Normalizer(clean_string=False,
                            abbreviation_file=os.path.join(TRAINING_DATA_PATH, ABBREVIATIONS_FILENAME))
    summarizer = DataSummarizer(input_file=os.path.join(TRAINING_DATA_PATH, CORPUS_FILENAME), normalizer=normalizer)
    summarizer.print_input_data_summary()
    print("Stats with string cleaning")
    normalizer = Normalizer(clean_string=True,
                            abbreviation_file=os.path.join(TRAINING_DATA_PATH, ABBREVIATIONS_FILENAME))
    summarizer = DataSummarizer(input_file=os.path.join(TRAINING_DATA_PATH, CORPUS_FILENAME), normalizer=normalizer)
    summarizer.print_input_data_summary()
