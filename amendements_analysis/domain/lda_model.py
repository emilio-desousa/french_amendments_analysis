import amendements_analysis.settings.base as stg
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords



class LDA_model:

    def __init__(self):

    


    @property
    def lda_model(self):



    def _stopwords_list(self):
        nltk.download('stopwords')
        stopwords_list = stopwords.words("french")
        stopwords_list = self._clean_stopwords_list(stopwords_list=stopwords_list)
        return stopwords_list

    def _clean_stopwords_list(self, stopwords_list):
        cleaned_stopwords_list = [word.encode('ascii', errors='ignore').decode('utf8') for word in stopwords_list]
        cleaned_stopwords_list = cleaned_stopwords_list.extend(stg.STOPWORDS_TO_ADD)
        return cleaned_stopwords_list
