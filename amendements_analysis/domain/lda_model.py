import amendements_analysis.settings.base as stg
from amendements_analysis.infrastructure.building_dataset import DatasetBuilder
from amendements_analysis.infrastructure.sentence_encoding import DatasetCleaner
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
import pyLDAvis
import pickle


class LatentDirichletAllocationModel:
    """
    Performs a Latent Dirichlet model to a list of amendments

    Attributes:
        amendments_list list: list of all amendments
    """

    def __init__(self, amendments_list):
        """
        Class initilization

        Parameters:
            amendments_list list: list of all amendments
        """
        self.amendments_list = amendments_list

    @property
    def lda_model(self):
        """
        Property to perform the LDA

        Returns:
            lda_model sklearn.decomposition.LatentDirichletAllocation: Model of LDA
        """
        tf, processed_amendments_list = self._data_preparation()
        lda = LatentDirichletAllocation(**stg.PARAMETERS_LDA).fit(
            processed_amendments_list
        )
        return lda, tf, processed_amendments_list

    def _data_preparation(self):
        stopwords_list = self._set_stopwords_list()
        tf = CountVectorizer(**stg.PARAMETERS_CV, stop_words=stopwords_list)
        processed_amendments_list = tf.fit_transform(self.amendments_list)
        return tf, processed_amendments_list

    def _set_stopwords_list(self):
        stopwords_list = stg.get_stopwords()
        stopwords_list = self._clean_stopwords_list(stopwords_list=stopwords_list)
        return stopwords_list

    def _clean_stopwords_list(self, stopwords_list):
        cleaned_stopwords_list = [
            word.encode("ascii", errors="ignore").decode("utf8")
            for word in stopwords_list
        ]
        cleaned_stopwords_list = cleaned_stopwords_list.extend(stg.STOPWORDS_TO_ADD)
        return cleaned_stopwords_list


if __name__ == "__main__":
    df = DatasetBuilder().data
    amendments_list = DatasetCleaner(df, partition=10).sentences
    lda, tf, processed_amendments_list = LatentDirichletAllocationModel(
        amendments_list
    ).lda_model
    vis = pyLDAvis.sklearn.prepare(
        lda, processed_amendments_list, tf, mds="tsne", R=10, sort_topics=True
    )
    pickle.dump(vis, open("viz.html", "wb"))
