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
    """
    Performs a Latent Dirichlet model to a list of amendments

    Attributes
    ----------
    amendments_list: list
    
    Properties
    ----------
    lda_model: sklearn.decomposition.LatentDirichletAllocation
    """

    def __init__(self, amendments_list):
        """Class initilization
        Parameters
        ----------
        amendments_list: list
        """
        self.amendments_list = amendments_list

    @property
    def lda_model(self):
        """Property to perform the LDA
        Returns
        -------
        lda_model: sklearn.decomposition.LatentDirichletAllocation
        """
        processed_amendments_list = self._data_preparation()
        lda = LatentDirichletAllocation(**stg.PARAMETERS_LDA).fit_transform(processed_amendments_list)
        return lda
    
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
    lda = LatentDirichletAllocationModel(amendments_list).lda_model