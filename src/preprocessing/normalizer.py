import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

from src.preprocessing.mentions.abbreviation_resolver import *
from src.utils.string_utils import *
from parser import *


class Normalizer:
    """
    Base class for normalizing into a format which aids in comparison
    """

    def __init__(self, abbreviation_file=None, clean_string=True):
        if abbreviation_file is None:
            abbreviation_file = os.path.join(TRAINING_DATA_PATH, ABBREVIATIONS_FILENAME)
        self.abbreviation_resolver = AbbreviationResolver.get_abbreviation_resolver(abbreviation_file)
        self.clean_string = clean_string

    def process(self, lookup_text, pmid=None):
        if pmid is not None:
            lookup_text = self.abbreviation_resolver.expand_abbreviation(pmid, lookup_text)
        long_form = re.sub(r' {2,}', " ", lookup_text.strip())
        if self.clean_string:
            processed_text = clean_text(long_form)
        else:
            processed_text = clean_text(text=long_form, lower_case=False, remove_punct=False, use_stem=False,
                                        remove_stop=False)
        return processed_text
