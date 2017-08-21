from nltk.tokenize import sent_tokenize
from config.data import *
import re
from src.utils.string_utils import *


class Entity:
    def __init__(self, mention, sdui_str, pmid, normalizer, type, omim_msh_mapping=None):
        self.pmid = pmid
        self.mention_str = str(mention)
        self.normalized_mention = self._preprocess(pmid, str(mention), normalizer)
        self.omim_msh_mapping = omim_msh_mapping
        self.sdui_str = sdui_str
        self.sdui_set = set([])
        self._parse_id(sdui_str)
        self.is_ambiguous = False
        self.type = type

    def _parse_id(self, sdui_str):
        """
        Process sdui string to gives a list of sdui
        :param sdui_str:
        :return: list of sdui's
        """
        if '|' in sdui_str:
            sdui_list = sdui_str.split('|')
        elif '+' in sdui_str:
            sdui_list = sdui_str.split('+')
        else:
            sdui_list = [sdui_str]
        for sdui in sdui_list:
            if sdui.startswith("OMIM"):
                if self.omim_msh_mapping is not None and sdui in self.omim_msh_mapping:
                    self.sdui_set.add(self.omim_msh_mapping[sdui])
                else:
                    if sdui not in self.omim_msh_mapping:
                        # print("No mapping found for %s" % sdui)
                        pass
            else:
                self.sdui_set.add(sdui)

    def _preprocess(self, pmid, mention, normalizer):
        """
        Using a preprocessor object convert mention to a form which is uniform across all mentions
        :param mention:
        :return:
        """
        if normalizer is not None:
            return normalizer.process(pmid=pmid, lookup_text=mention)
        else:
            return mention

    def merge(self, another_entity):
        """
        Merge two entities together if they have same mention
        :param another_entity:
        :return:
        """
        if self.normalized_mention != another_entity.normalized_mention:
            return False
        else:
            if self.is_ambiguous or another_entity.is_ambiguous:
                self.is_ambiguous = True
            elif self.sdui_set != another_entity.sdui_set:
                self.is_ambiguous = True
            self.sdui_set = self.sdui_set.union(another_entity.sdui_set)
            return True


class Sentence:
    """
    A model  class representing a sentence containing a mention which is surrounded by <START> and <END> tokens
    """

    def __init__(self, text, entity):
        self.text = text
        self.entity = entity
        self.left_component = self.get_left_component()
        self.right_component = self.get_right_component()

    def get_left_component(self):
        index = self.text.find(MENTION_END)
        left_sentence = self.text[:index]
        if MENTION_END in left_sentence:
            raise Exception("Some problem in %s as there should be only single tag" % self.text)
        return self.clean_tags(left_sentence)

    def get_right_component(self):
        index = self.text.find(MENTION_START) + len(MENTION_START)
        right_sentence = self.text[index:]
        if MENTION_START in right_sentence:
            raise Exception("Some problem in %s as there should be only single tag" % self.text)
        return self.clean_tags(right_sentence)

    @staticmethod
    def clean_tags(text):
        text = re.sub(r'%s' % MENTION_START, '', text)
        text = re.sub(r'%s' % MENTION_END, '', text)
        text = re.sub(r' {2,}', ' ', text)
        return text

    @staticmethod
    def convert_to_sentences(text, entity_list):
        sentences = sent_tokenize(text)
        out = []
        previous_sentence = ""
        entity_index = 0
        for sentence in sentences:
            sentence = previous_sentence + ' ' + sentence
            if MENTION_START in sentence or MENTION_END in sentence:
                if MENTION_END not in sentence:
                    previous_sentence = sentence
                elif MENTION_START not in sentence:
                    raise Exception("Incorrect tokenization of sentence: %s" % sentence)
                else:
                    # This sentence contains balanced tags.
                    current = 0
                    while sentence.find(MENTION_START, current) >= 0:
                        start_index = sentence.find(MENTION_START, current)
                        end_index = sentence.find(MENTION_END, start_index)
                        mention = sentence[start_index + len(MENTION_START) + 1:end_index - 1]
                        current = end_index + len(MENTION_END)
                        if compare_mentions(mention, entity_list[entity_index].mention_str):
                            clean_sentence = Sentence.clean_tags(sentence[:start_index]) + sentence[
                                                                                           start_index:current] + Sentence.clean_tags(
                                sentence[current:])
                            out.append(Sentence(clean_sentence, entity_list[entity_index]))
                            entity_index += 1
                            previous_sentence = ""
                        else:
                            raise Exception(
                                "Some problem with the entity matching for text :%s and line:%s : Expected mention: %s and found %s " % (
                                    text, sentence, mention, entity_list[entity_index].mention_str))
        return out
