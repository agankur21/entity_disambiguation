import copy

from data_cleaning import *
from src.models.mention import *
from src.utils.string_utils import *


class PubtatorDataEntityMentions(object):
    def __init__(self, normalizer=None):
        self.pmid_entity_map = {}
        self.entity_list = []
        self.merged_entity_list = []
        self.sdui_map = {}
        self.normalizer = normalizer
        self.omim_mesh_mappings = get_OMIM_MSH_mapping()

    def parse_entities(self, line, filter_pmids=set([])):
        """
        Parse a pubtator line containing mention information : Typically it is a 6 token tab separated line
        :param line:
        :return:
        """
        line = line.strip()
        if len(line) > 0:
            split_array = line.split('\t')
            if len(split_array) >= 6:
                # Line contains Entity mention
                pmid = split_array[0].strip()
                if len(filter_pmids) > 0 and pmid not in filter_pmids:
                    return
                sdui_str = split_array[-1].strip()
                entity_type = split_array[-2].strip()
                mention = split_array[-3].strip()
                entity = Entity(mention=mention, sdui_str=sdui_str, pmid=pmid, normalizer=self.normalizer,
                                type=entity_type, omim_msh_mapping=self.omim_mesh_mappings)
                if pmid not in self.pmid_entity_map:
                    self.pmid_entity_map[pmid] = []
                self.pmid_entity_map[pmid].append(entity)
                self.entity_list.append(entity)

    @staticmethod
    @raise_exception
    def merge_entities(list_entities):
        """
        Since there may be many duplicate entities with similar mention, this function merges them
        :param list_entities:
        :return:
        """
        mention_map = {}
        for entity in list_entities:
            if entity.normalized_mention not in mention_map:
                mention_map[entity.normalized_mention] = copy.deepcopy(entity)
            else:
                if not mention_map[entity.normalized_mention].merge(entity):
                    raise RuntimeError
        return mention_map.values()

    def update_sdui_map(self):
        """
        Update the reverse map of SDUI -> Set of Entity Mentions associated with it
        :return:
        """
        for entity in self.entity_list:
            for sdui in entity.sdui_set:
                if sdui not in self.sdui_map:
                    self.sdui_map[sdui] = set([])
                self.sdui_map[sdui].add(entity.normalized_mention)

    @staticmethod
    def get_pmids(mode, directory):
        if mode == 'all':
            return []
        elif mode == 'train_dev':
            return filter(lambda x: len(x.strip()) > 0,
                          open(os.path.join(directory, PMID_FILE_TRAIN_DEV), 'r').read().splitlines())
        elif mode == 'train':
            if os.path.isfile(os.path.join(directory, PMID_FILE_TRAIN)):
                return filter(lambda x: len(x.strip()) > 0,
                              open(os.path.join(directory, PMID_FILE_TRAIN), 'r').read().splitlines())
            else:
                return []
        elif mode == 'dev':
            if os.path.isfile(os.path.join(directory, PMID_FILE_DEV)):
                return filter(lambda x: len(x.strip()) > 0,
                              open(os.path.join(directory, PMID_FILE_DEV), 'r').read().splitlines())
            else:
                return []
        elif mode == 'test':
            if os.path.isfile(os.path.join(directory, PMID_FILE_TEST)):
                return filter(lambda x: len(x.strip()) > 0,
                              open(os.path.join(directory, PMID_FILE_TEST), 'r').read().splitlines())
            else:
                return []
        else:
            return []

    @staticmethod
    def parse_pubtator_data(file_path, normalizer=None, mode='all'):
        """
        Parse the file in pubtator format and populate the required fields
        :param file_path:
        :param normalizer
        :param mode
        :return: An object of PubtatorDataEntityMentions with all fields populated
        """
        filter_pmids = set(PubtatorDataEntityMentions.get_pmids(mode, os.path.dirname(file_path)))
        pubtatordata = PubtatorDataEntityMentions(normalizer=normalizer)
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                pubtatordata.parse_entities(line, filter_pmids)
        pubtatordata.merged_entity_list = pubtatordata.merge_entities(pubtatordata.entity_list)
        pubtatordata.update_sdui_map()
        return pubtatordata


class PubtatorData(PubtatorDataEntityMentions):
    def __init__(self, normalizer=None):
        super(PubtatorData, self).__init__(normalizer)
        self.pmid_sentence_map = {}

    def parse_text(self, line, pmid_text_map, flag='t'):
        line_array = line.split('\t')
        if len(line.strip()) == 0 or len(line_array) >= 2:
            return
        else:
            elements = line.split('|')
            if len(elements) >= 3 and elements[1] == flag:
                if elements[0] not in pmid_text_map:
                    if flag == 't':
                        pmid_text_map[elements[0]] = {}
                    else:
                        raise Exception("Some error in parsing PMID for line %s" % line)
                pmid_text_map[elements[0]][flag] = elements[2]

    def parse_offsets(self, line, pmid_text_map):
        line_array = line.split('\t')
        if len(line.strip()) > 0 and len(line_array) >= 6:
            if line_array[0] not in pmid_text_map:
                raise Exception("Some error in parsing PMID for line %s" % line)
            else:
                if 'list_offsets' not in pmid_text_map[line_array[0]]:
                    pmid_text_map[line_array[0]]['list_offsets'] = []
                pmid_text_map[line_array[0]]['list_offsets'].append(
                    (int(line_array[1]), int(line_array[2]), line_array[3]))

    @staticmethod
    def parse_pubtator_data(file_path, normalizer=None, mode='all'):
        """
        Parse the file in pubtator format and populate the required fields
        :param file_path:
        :param normalizer:
        :param mode:
        :return:
        """
        filter_pmids = set(PubtatorData.get_pmids(mode, os.path.dirname(file_path)))
        pubtatordata = PubtatorData(normalizer=normalizer)
        pmid_text_map = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                pubtatordata.parse_text(line, pmid_text_map, 't')
                pubtatordata.parse_text(line, pmid_text_map, 'a')
                pubtatordata.parse_offsets(line, pmid_text_map)
                pubtatordata.parse_entities(line, filter_pmids)
        if len(filter_pmids) == 0:
            filter_pmids = pmid_text_map.keys()
        for pmid in filter_pmids:
            if pmid not in pmid_text_map:
                print "WARNING: PMID - %s in filter file is not present in Corpus" % pmid
                continue
            text = pmid_text_map[pmid]['t'] + ' ' + pmid_text_map[pmid]['a']
            modified_text_list = []
            offset = 0
            for start, end, mention in pmid_text_map[pmid]['list_offsets']:

                if not compare_mentions(mention, text[start:end]):
                    raise Exception("Some error in the offsets for pmid: %s as expected mention: %s and found: %s" % (
                        pmid, mention, text[start:end]))
                modified_text_list.append(text[offset:start])
                modified_text_list.append(' ' + MENTION_START + ' ')
                modified_text_list.append(mention)
                modified_text_list.append(' ' + MENTION_END + ' ')
                offset = end
            modified_text_list.append(text[offset:])
            pubtatordata.pmid_sentence_map[pmid] = Sentence.convert_to_sentences("".join(modified_text_list),
                                                                                 pubtatordata.pmid_entity_map[pmid])
        return pubtatordata


if __name__ == '__main__':
    pubtator_data_train = PubtatorDataEntityMentions.parse_pubtator_data(os.path.join(TRAINING_DATA_PATH, "Corpus.txt"),
                                                                         mode='train')
    pubtator_data_test = PubtatorDataEntityMentions.parse_pubtator_data(os.path.join(TRAINING_DATA_PATH, "Corpus.txt"),
                                                                        mode='test')
    train_sdui_set = set(pubtator_data_train.sdui_map.keys())
    total_num = len(pubtator_data_test.entity_list)
    mismatch = 0
    for entity in pubtator_data_test.entity_list:
        if len(entity.sdui_set.intersection(train_sdui_set)) == 0:
            mismatch += 1
    print 1.0 * mismatch / total_num
