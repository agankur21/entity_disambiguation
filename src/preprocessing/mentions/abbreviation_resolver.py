import re

from config.data import *
from src.utils.decorator_utils import *


class AbbreviationResolver:
    def __init__(self):
        self.abbreviations = {}

    @raise_exception
    def load_abbreviations(self, filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    split_line = line.split("\t")
                    self.create_abbreviation(split_line[0], split_line[1], split_line[2])

    @raise_exception
    def create_abbreviation(self, pmid, short_form, long_form):
        if pmid not in self.abbreviations:
            self.abbreviations[pmid] = {short_form: long_form}
        else:
            if short_form in self.abbreviations[pmid]:
                if self.abbreviations[pmid][short_form] != long_form:
                    raise ValueError
            else:
                self.abbreviations[pmid][short_form] = long_form

    def expand_abbreviation(self, pmid, lookup_text):
        if pmid not in self.abbreviations:
            return lookup_text
        else:
            return self.retrieve_long_form(lookup_text, self.abbreviations[pmid])

    @staticmethod
    def retrieve_long_form(lookup_text, abbreviation_dictionary):
        if abbreviation_dictionary is None:
            return lookup_text
        else:
            for abbreviation in abbreviation_dictionary:
                if abbreviation in lookup_text:
                    long_form = abbreviation_dictionary[abbreviation]
                    """
                    2 Cases
                     - Only short form
                     - Both short form and long form
                    """
                    if long_form in lookup_text:
                        lookup_text = re.sub(r'\(?' + re.escape(abbreviation) + r'\)?', "", lookup_text)
                    else:
                        lookup_text = re.sub(r'\(?' + re.escape(abbreviation) + r'\)?', long_form, lookup_text)
            return lookup_text

    @staticmethod
    @raise_exception
    def get_abbreviation_resolver(abbreviation_file_path):
        resolver = AbbreviationResolver()
        if abbreviation_file_path is None or not os.path.isfile(abbreviation_file_path):
            resolver.load_abbreviations(os.path.join(TRAINING_DATA_PATH, ABBREVIATIONS_FILENAME))
        else:
            resolver.load_abbreviations(abbreviation_file_path)
        return resolver
