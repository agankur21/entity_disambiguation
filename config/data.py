import os

###############NCBI DATA CORPUS######################################
TRAINING_DATA_PATH = "/Users/aaggarwal/Documents/Course/CZI/EntityLinking/entity_linking/data/ncbi_disease_corpus/raw_text"
# TRAINING_DATA_PATH = "/Users/aaggarwal/Documents/ner_data/raw_data/cdr_pubtator"
ABBREVIATIONS_FILENAME = "abbreviations.tsv"
CORPUS_FILENAME = "Corpus.txt"
HOME_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(HOME_DIR, "data")
MRCONSO_INDEXED_DATA = "mrconso.db"
PMID_FILE_TRAIN_DEV = "NCBI_corpus_training_development_PMIDs.txt"
PMID_FILE_TRAIN = "NCBI_corpus_training_PMIDs.txt"
PMID_FILE_DEV = "NCBI_corpus_development_PMIDs.txt"
PMID_FILE_TEST = "NCBI_corpus_test_PMIDs.txt"

###############UMLS DATABASE #########################################
MRCONSO_AUI_INDEX = 7
MRCONSO_CUI_INDEX = 0
MRCONSO_STR_INDEX = 14
MRCONSO_SDUI_INDEX = 10
MRCONSO_CODE_INDEX = 13
MRCONSO_SAB_INDEX = 11
MRREL_CUI1_INDEX = 0
MRREL_AUI1_INDEX = 1
MRREL_STYPE1_INDEX = 2
MRREL_REL_INDEX = 3
MRREL_CUI2_INDEX = 4
MRREL_AUI2_INDEX = 5
MRREL_STYPE2_INDEX = 6
MRREL_SAB_INDEX = 10

################################### DATA CLEANING############################
RELATION_PRIORITY = ["Parent", "Child", "Sibling", "Broad", "Narrow", "Qualifier", "Allowed qualifier", "Other"]
MENTION_START = "<MENTION_START>"
MENTION_END = "<MENTION_END>"
ZERO_STR = "<ZERO>"
PAD_STR = "<PAD>"
OOV_STR = "<OOV>"
NONE_STR = "<NONE>"
SENT_START = "<S>"
SENT_END = "</S>"
