import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from src.preprocessing.string_similarity import StringSimilarity
from src.preprocessing.normalizer import *
from src.utils.string_utils import *
import numpy as np


def tune_fuzzy_matching_params():
    strsim = StringSimilarity()
    indexed_folder = os.path.join(DATA_DIR, "umls", "indexed")
    db_name = "concepts.db"
    normalizer = Normalizer(clean_string=True,
                            abbreviation_file=os.path.join(TRAINING_DATA_PATH, ABBREVIATIONS_FILENAME))
    pubtator_data = PubtatorDataEntityMentions.parse_pubtator_data(
        file_path=os.path.join(TRAINING_DATA_PATH, CORPUS_FILENAME), normalizer=normalizer, mode="all")
    for measure in [simstring.overlap, simstring.cosine, simstring.jaccard, simstring.exact]:
        for threshold in np.arange(0.4, 1.1, 0.1):
            strsim.update_retrieval_params(measure, threshold)
            total_entity_score, distinct_entity_score = strsim.match_database(pubtator_data,
                                                                              db_directory=indexed_folder,
                                                                              db_name=db_name)
            print "%0.3f%s of total valid entities have correct SDUI's captured with measure: %d and threshold : %f " % (
                total_entity_score, "%", measure, threshold)
            print "%0.3f%s of total distinct valid entities have correct SDUI's captured with measure: %d and threshold : %f " % (
                distinct_entity_score, "%", measure, threshold)


if __name__ == '__main__':
    tune_fuzzy_matching_params()
