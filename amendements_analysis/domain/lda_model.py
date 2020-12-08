import amendements_analysis.settings.base as stg
from amendements_analysis.infrastructure.building_dataset import DatasetBuilder
from amendements_analysis.infrastructure.sentence_encoding import DatasetCleaner
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords



class LatentDirichletAllocationModel:

    def __init__(self, amendments_list):
        self.amendments_list = amendments_list

    @property
    def lda_model(self):
        processed_amendments_list = self._data_preparation()
        processed_amendments_list, _ = train_test_split(cleaned_amendments_list, test_size=0.9, shuffle=True)
        lda = LatentDirichletAllocation(**stg.PARAMETERS_LDA).fit_transform(cleaned_amendments_list)
        return 
    
    def _data_preparation(self):
        stopwords_list = self._set_stopwords_list()
        processed_amendments_list = CountVectorizer(**stg.PARAMETERS_CV, stop_words=stopwords_list)\
                                    .fit_transform(self.amendments_list)
        return processed_amendments_list

    def _set_stopwords_list(self):
        nltk.download('stopwords')
        stopwords_list = stopwords.words("french")
        stopwords_list = self._clean_stopwords_list(stopwords_list=stopwords_list)
        return stopwords_list

    def _clean_stopwords_list(self, stopwords_list):
        cleaned_stopwords_list = [word.encode('ascii', errors='ignore').decode('utf8') for word in stopwords_list]
        cleaned_stopwords_list = cleaned_stopwords_list.extend(stg.STOPWORDS_TO_ADD)
        return cleaned_stopwords_list


if __name__ == "__main__":
    df = DatasetBuilder().data
    amendments_list = DatasetCleaner(df, partition=10).sentences
    print(amendments_list)