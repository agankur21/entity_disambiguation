from parser import *
from src.models.umls import *
from src.preprocessing.normalizer import Normalizer


class StringSimilarity:
    def __init__(self):
        self.measure = simstring.overlap
        self.threshold = 0.65

    def update_retrieval_params(self, measure, threshold):
        self.measure = measure
        self.threshold = threshold

    def create_normalized_database(self, list_input_files, db_directory, db_name, normalizer):
        indexed_data_connector = SimStringConnector(directory=DATA_DIR,
                                                    filename=os.path.join(db_directory, db_name),
                                                    mode='write')
        for file in list_input_files:
            with open(file, 'r') as f:
                for line in f:
                    if is_ascii(line.strip()):
                        mrconso = MRConso.parse(line.strip(), '|', normalizer)
                        if mrconso is not None and mrconso.sdui is not "" and mrconso.processed_str is not "":
                            new_line = "%s|%s" % (mrconso.sdui, mrconso.processed_str)
                            indexed_data_connector.insert(new_line)
        indexed_data_connector.writer.close()

    def match_database(self, pubtator_data, db_directory, db_name):
        indexed_data_connector = SimStringConnector(directory=db_directory, filename=db_name, mode='read',
                                                    measure=self.measure, threshold=self.threshold)
        num_mention_match_sdui_match = 0
        total_valid_mentions = 0
        for entity in pubtator_data.entity_list:
            if len(entity.sdui_set) > 0:
                total_valid_mentions += 1
                set_predictions = set(
                    [x.split("|")[0].strip() for x in list(indexed_data_connector.retrieve(entity.normalized_mention))])
                if len(set_predictions.intersection(entity.sdui_set)) > 0:
                    num_mention_match_sdui_match += 1
        total_entity_score = 1.0 * num_mention_match_sdui_match / total_valid_mentions
        num_mention_match_sdui_match = 0
        total_valid_mentions = 0
        for entity in pubtator_data.merged_entity_list:
            if len(entity.sdui_set) > 0:
                total_valid_mentions += 1
                set_predictions = set(
                    [x.split("|")[0].strip() for x in list(indexed_data_connector.retrieve(entity.normalized_mention))])
                if len(set_predictions.intersection(entity.sdui_set)) > 0:
                    num_mention_match_sdui_match += 1
        distinct_entity_score = 1.0 * num_mention_match_sdui_match / total_valid_mentions
        return total_entity_score, distinct_entity_score


if __name__ == '__main__':
    strsim = StringSimilarity()
    raw_text_file = '/Users/aaggarwal/Documents/Course/CZI/EntityLinking/entity_linking/data/umls/umls_concepts.txt'
    indexed_folder = "/Users/aaggarwal/Documents/Course/CZI/EntityLinking/entity_linking/data/umls/indexed"
    db_name = "concepts.db"
    normalizer = Normalizer(clean_string=True,
                            abbreviation_file=os.path.join(TRAINING_DATA_PATH, ABBREVIATIONS_FILENAME))
    pubtator_data = PubtatorDataEntityMentions.parse_pubtator_data(
        file_path=os.path.join(TRAINING_DATA_PATH, CORPUS_FILENAME), normalizer=normalizer)
    # strsim.create_normalized_database([raw_text_file], indexed_folder, db_name, normalizer)
    print strsim.match_database(pubtator_data, db_directory=indexed_folder,
                                db_name=db_name)
