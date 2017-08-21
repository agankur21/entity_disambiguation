import os
import re
import simstring
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from config.data import *

stemmer = PorterStemmer()
from decorator_utils import *


class SimStringConnector:
    @raise_exception
    def __init__(self, directory, filename, measure=simstring.overlap, threshold=0.65, mode='write'):
        if not (filename.endswith('.db') and os.path.isdir(directory)):
            raise ValueError("Incorrect file format for Database. Database must end with .db")
        else:
            self.writer = None
            self.reader = None
            if mode == 'write':
                self.writer = simstring.writer(os.path.join(directory, filename))
            else:
                self.reader = simstring.reader(os.path.join(directory, filename))
                self.reader.measure = measure
                self.reader.threshold = threshold

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
        if self.reader is not None:
            self.reader.close()

    @raise_exception
    def insert(self, line):
        if is_ascii(line):
            line = str(line)
            self.writer.insert(line)

    @raise_exception
    def retrieve(self, text):
        if is_ascii(text):
            text = str(text)
            return self.reader.retrieve(text)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.reader is not None:
            self.reader.close()


stop_words = set(["a", "an", "and", "are", "as", "at", "be", "but", "by",
                  "for", "if", "in", "into", "is", "it",
                  "no", "not", "of", "on", "or", "such",
                  "that", "the", "their", "then", "there", "these",
                  "they", "this", "to", "was", "will", "with"])

nltk_stop_words = set(stopwords.words('english'))

stop_words = stop_words.union(nltk_stop_words)

punctuation = """!"#$%&'()*+,./:;<=>?@[\]^_`{|}~"""

NOT_AVAILABLE = "N/A"


def is_digit(s):
    return all(47 < ord(c) < 58 for c in s)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def remove_punctuation(text):
    return text.translate(None, punctuation)


def remove_double_spaces(text):
    return re.sub(r" {2,}", " ", text)


def remove_trailing_space(text):
    return text.strip()


@raise_exception
def stem(text):
    stemmed_tokens = []
    try:
        tokens = text.split()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
    except UnicodeDecodeError:
        pass
    return " ".join(stemmed_tokens)


@raise_exception
def remove_stopwords(text):
    tokens = text.split()
    out = []
    for token in tokens:
        if token not in stop_words:
            out.append(token)
    return " ".join(out)


def add_space(text, identifier):
    out = []
    for c in text:
        out.append(c)
        if c == identifier:
            out.append(' ')
    return "".join(out)


def clean_text(text, lower_case=True, remove_punct=True, use_stem=True, remove_stop=True):
    """
    Clean the text by converting to lower case, removing stop words,removing punctuation and stemming
    :param text:
    :param lower_case:
    :param remove_punct:
    :param use_stem:
    :param remove_stop:
    :return:
    """
    if lower_case:
        text = text.lower()
    text = add_space(text, ',')
    if remove_punct:
        text = remove_punctuation(text)
    if remove_stop:
        text = remove_stopwords(text)
    text = remove_trailing_space(remove_double_spaces(text))
    if use_stem:
        text = stem(text)
    return text


def compare_mentions(text1, text2):
    text1 = re.sub("\"", " ", text1)
    text2 = re.sub("\"", " ", text2)
    return text1.strip() == text2.strip()


if __name__ == '__main__':
    # simstringconnector = SimStringConnector(DATABASE_PATH, "test.db", mode="write")
    # simstringconnector.insert(
    #     "C0000005|ENG|P|L0000005|PF|S0007492|Y|A26634265||M0019694|D012711|MSH|PEP|D012711|(131)I-Macroaggregated Albumin|0|N|256|")
    # simstringconnector.insert(
    #     "C0000039|CZE|P|L6742182|PF|S7862052|Y|A13042554||M0023172|D015060|MSHCZE|MH|D015060|1,2-dipalmitoylfosfatidylcholin|3|N||")
    # simstringconnector.close()
    # simstringconnector = SimStringConnector(DATABASE_PATH, "test.db", mode="read", measure=simstring.overlap,
    #                                         threshold=0.65)
    # print simstringconnector.retrieve("Albumin")
    # simstringconnector.close()
    print stem("tumour")
    print stem("tumor")
