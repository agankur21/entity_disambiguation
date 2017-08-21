from org.apache.lucene.analysis import LowerCaseFilter, StopFilter
from org.apache.lucene.analysis.core import StopAnalyzer
from org.apache.lucene.analysis.en import PorterStemFilter
from org.apache.lucene.analysis.standard import StandardTokenizer, StandardFilter
from org.apache.pylucene.analysis import PythonAnalyzer


class DiseaseNameAnalyzer(PythonAnalyzer):

    def createComponents(self, fieldName):
        source = StandardTokenizer()
        filter = StandardFilter(source)
        filter = LowerCaseFilter(filter)
        filter = PorterStemFilter(filter)
        filter = StopFilter(filter, StopAnalyzer.ENGLISH_STOP_WORDS_SET)
        return self.TokenStreamComponents(source, filter)

    def initReader(self, fieldName, reader):
        return reader


