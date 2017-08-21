from src.utils.multiprocess_utils import *


class AbbreviationIdentifier:
    @raise_exception
    def __init__(self, processed_file, abbreviation_file):
        self.input_processed_file = open(processed_file,
                                         'r')  # The file contains each followed by abbreviation in the next line
        self.abbreviation_file = open(abbreviation_file, 'w')  # Output file containing all the abbreviations
        self.get_abbreviations()

    def __del__(self):
        self.input_processed_file.close()
        self.abbreviation_file.close()

    @timeit
    @raise_exception
    def get_abbreviations(self):
        pmid = None
        for line in self.input_processed_file:
            if '|t|' in line:
                pmid = line.strip().split('|')[0]
            elif line.startswith("  ") and '|' in line:
                split_line = line.strip().split("|")
                if len(split_line) == 3:
                    print("Found abbreviaiton pair: %s -> %s for pmid : %s" % (split_line[0], split_line[1], pmid))
                    self.abbreviation_file.write("%s\t%s\t%s\n" % (pmid, split_line[0], split_line[1]))
                elif len(split_line) == 5:
                    print("Found abbreviaiton pair: %s -> %s for pmid : %s" % (split_line[0], split_line[3], pmid))
                    self.abbreviation_file.write("%s\t%s\t%s\n" % (pmid, split_line[0], split_line[3]))


if __name__ == '__main__':
    processed_file = '/Users/aaggarwal/Documents/ner_data/raw_data/cdr_pubtator/CDR_TrainingSet.PubTator_abbrev.txt'
    abbrev_file = "/Users/aaggarwal/Documents/ner_data/raw_data/cdr_pubtator/abbreviations.txt"
    abb_identifier = AbbreviationIdentifier(processed_file, abbrev_file)
