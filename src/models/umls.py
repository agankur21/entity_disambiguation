from src.utils.string_utils import *

class MRConso:
    def __init__(self, cui=None, lat=None, ts=None, lui=None, stt=None, sui=None, ispref=None, aui=None, saui=None,
                 sdui=None, sab=None, tty=None, code=None, str=None, normalizer=None):
        self.cui = cui
        self.lat = lat
        self.ts = ts
        self.lui = lui
        self.stt = stt
        self.sui = sui
        self.ispref = ispref
        self.aui = aui
        self.saui = saui
        self.sdui = sdui
        self.sab = sab
        self.tty = tty
        self.code = code
        self.str = str
        self.normalizer = normalizer
        self.processed_str = self._preprocess(self.str)

    def _preprocess(self, str):
        """
        Using a preprocessor object convert mention to a form which is uniform across all mentions
        :param mention:
        :return:
        """
        if self.normalizer is not None and str is not None:
            return self.normalizer.process(lookup_text=str)
        else:
            return str

    @staticmethod
    def parse(line, delimiter, normalizer):
        line = line.strip()
        line_array = line.split(delimiter)
        if is_ascii(line) and len(line_array) >= 18:
            mrconso = MRConso(cui=line_array[MRCONSO_CUI_INDEX], sdui=line_array[MRCONSO_SDUI_INDEX],
                              aui=line_array[MRCONSO_AUI_INDEX],
                              str=line_array[MRCONSO_STR_INDEX], code=line_array[MRCONSO_CODE_INDEX],
                              normalizer=normalizer)
        else:
            mrconso = None
        return mrconso

    def to_str(self, delimiter):
        return self.sdui + delimiter + self.processed_str

    @staticmethod
    def parse_database(input_file, normalizer):
        out = []
        with open(input_file, 'r') as f:
            for line in f:
                mrconso = MRConso.parse(line, "|", normalizer)
                if mrconso is not None:
                    out.append(mrconso)
        return out
