from config.data import *
import csv

from src.utils.string_utils import is_ascii


def get_aui_sdui_mapping(input_file=None):
    """
    Get the AUI to SDUI mapping from MRCONSO table. Since AUI is a primary key.
    :return:
    """
    input_file = os.path.join(DATA_DIR, "umls", "MRCONSO.RRF") if input_file is None else input_file
    mapping = {}
    with open(input_file, 'r') as f:
        for line in f:
            line_array = line.split("|")
            if line_array[MRCONSO_SAB_INDEX] == 'MSH' and line_array[MRCONSO_SDUI_INDEX].strip() != "":
                mapping[line_array[MRCONSO_AUI_INDEX]] = line_array[MRCONSO_SDUI_INDEX]
    return mapping


def get_aui_str_mapping(input_file=None):
    """
    Get the AUI to str mapping from MRCONSO table. Since AUI is a primary key.
    :return:
    """
    input_file = os.path.join(DATA_DIR, "umls", "MRCONSO.RRF") if input_file is None else input_file
    mapping = {}
    with open(input_file, 'r') as f:
        for line in f:
            line_array = line.split("|")
            if line_array[MRCONSO_SAB_INDEX] == 'MSH' and line_array[MRCONSO_SDUI_INDEX].strip() != "":
                mapping[line_array[MRCONSO_AUI_INDEX]] = line_array[MRCONSO_STR_INDEX]
    return mapping


def clean_relations_data(input_file=None, out_file=None):
    """
    Create a complete relation file with the following details CUI1,AUI1,SDUI1,CUI2,AUI2,SDUI2,REL
    :return:
    """
    input_file = os.path.join(DATA_DIR, "umls", "MRREL.RRF") if input_file is None else input_file
    out_file = os.path.join(DATA_DIR, "umls", "umls_relations.txt") if out_file is None else out_file
    aui_sdui_mappings = get_aui_sdui_mapping()
    aui_str_mapping = get_aui_str_mapping()
    relations_mapping = get_relations_map()
    out = open(out_file, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            line_array = line.split("|")
            if line_array[MRREL_SAB_INDEX] == "MSH" and line_array[MRREL_STYPE1_INDEX] == "SDUI" and \
                            line_array[MRREL_STYPE2_INDEX] == "SDUI":
                try:
                    out_line = "%s|%s|%s|%s|%s|%s|%s|%s|%s\n" % (
                        line_array[MRREL_CUI1_INDEX], line_array[MRREL_AUI1_INDEX],
                        aui_sdui_mappings[line_array[MRREL_AUI1_INDEX]], aui_str_mapping[line_array[MRREL_AUI1_INDEX]],
                        line_array[MRREL_CUI2_INDEX], line_array[MRREL_AUI2_INDEX],
                        aui_sdui_mappings[line_array[MRREL_AUI2_INDEX]], aui_str_mapping[line_array[MRREL_AUI2_INDEX]],
                        relations_mapping[line_array[MRREL_REL_INDEX]])
                    out.write(out_line)
                except KeyError:
                    print "No mappings found for AUI in line : %s" % line
    out.close()


def clean_concepts_data(input_file=None, out_file=None):
    input_file = os.path.join(DATA_DIR, "umls", "MRCONSO.RRF") if input_file is None else input_file
    out_file = os.path.join(DATA_DIR, "umls", "umls_concepts.txt") if out_file is None else out_file
    out = open(out_file, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            line_array = line.split("|")
            if is_ascii(line) and line_array[MRCONSO_SAB_INDEX] == "MSH" and line_array[
                MRCONSO_SDUI_INDEX].strip() != "":
                out.write(line)
    out.close()


def get_mention_id_mappings(input_file_name, out_file, field_threshold, column_indices):
    """
    Write certain columns from input file to output file
    :param input_file_name:
    :param out_file:
    :param field_threshold:
    :param column_indices:
    :return:
    """
    out = open(out_file, 'w')
    if input_file_name.endswith('.csv'):
        with open(input_file_name, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) >= field_threshold:
                    if not line[0].startswith("#"):
                        row = []
                        for index in column_indices:
                            row.append(line[index])
                        out.write("|".join(row) + "\n")
    out.close()


def get_OMIM_MSH_mapping(input_file_name=None):
    """
    From the CTD database get the OMIM-> MESH embeddings whereever possible
    :param input_file_name:
    :return:
    """
    input_file_name = os.path.join(DATA_DIR, "ctd", "CTD_diseases.csv") if input_file_name is None else input_file_name
    mappings = {}
    with open(input_file_name, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) >= 9 and not line[0].startswith("#") and line[1].startswith("MESH:") \
                    and line[2].strip() != "" and "OMIM" in line[2]:
                omim_array = [x for x in line[2].split("|") if x.startswith("OMIM")]
                msh_id = line[1].split("MESH:")[1]
                for omim in omim_array:
                    if omim in mappings and mappings[omim] != msh_id:
                        print "Two incompatible mappings of OMIM : %s to MESH Id: %s and MESH Id :%s" % (
                            omim, mappings[omim], msh_id)
                    else:
                        mappings[omim] = msh_id
    return mappings


def get_relations_map(relations_map_file=None):
    if relations_map_file is None:
        relations_map_file = os.path.join(DATA_DIR, 'umls', 'relations_map.txt')
    out_map = {}
    with open(relations_map_file, 'r') as f:
        for l in f:
            l = l.strip()
            if len(l) > 0:
                abbrev, full_name = l.split("|")
                out_map[abbrev] = full_name
    return out_map


def identifying_bidirectional_relations(relations_file=None, out_file_path=None):
    if relations_file is None:
        relations_file = os.path.join(DATA_DIR, "umls", "umls_relations.txt")
        entity_pair_map = {}
        with open(relations_file, 'r') as f:
            for line in f:
                line = line.strip()
                line_array = line.split("|")
                e1, e2, rel = line_array[2], line_array[6], line_array[8]
                if (e1, e2) not in entity_pair_map and (e2, e1) not in entity_pair_map:
                    entity_pair_map[(e1, e2)] = set([])
                if (e2, e1) in entity_pair_map:
                    entity_pair_map[(e2, e1)].add(rel)
                else:
                    entity_pair_map[(e1, e2)].add(rel)
        if out_file_path is None:
            out_file_path = os.path.join(DATA_DIR, "umls", "unique_relations.txt")
        out_file = open(out_file_path, 'w')
        for entity_pair, relation_set in entity_pair_map.iteritems():
            out_file.write("%s\t%s\t%s\n" % (entity_pair[0], entity_pair[1], ",".join(relation_set)))
        out_file.close()


def apply_relation_priority(rel_set):
    relations = set(rel_set.split(","))
    for relation in RELATION_PRIORITY:
        if relation in relations:
            return relation


def prioritize_relations(relations_file=None, out_file_path=None):
    if relations_file is None:
        relations_file = os.path.join(DATA_DIR, "umls", "unique_relations.txt")
        if out_file_path is None:
            out_file_path = os.path.join(DATA_DIR, "umls", "kg.txt")
        out_file = open(out_file_path, 'w')
        with open(relations_file, 'r') as f:
            for line in f:
                line = line.strip()
                line_array = line.split("\t")
                [e1, e2, rel_set] = line_array
                rel = apply_relation_priority(rel_set)
                out_file.write("%s\t%s\t%s\t1\n" % (e1, e2, rel))
        out_file.close()


if __name__ == '__main__':
    # UMLS relations data
    # input_relations = os.path.join(DATA_DIR, "umls", "MRREL.RRF")
    # output_relations = os.path.join(DATA_DIR, "umls", "umls_relations.txt")
    # clean_relations_data(input_relations, output_relations)
    # clean_concepts_data()
    # CTD
    # get_OMIM_MSH_mapping()
    #identifying_bidirectional_relations()
    prioritize_relations()
pass
