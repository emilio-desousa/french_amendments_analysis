import amendements_analysis.settings.base as stg
import numpy as np
import pandas as pd
import os
import re
import spacy
import unicodedata, unidecode
from sklearn.feature_extraction.text import CountVectorizer

import fr_core_news_md


class TextCleaner:
    """
    Get amendments df and perform pre processing compatible with Topics words finder
    This includes :
        - remove accent
        - lemmatize
        - specific regex flag for specific terms

    Attributes
    ----------
    df: pandas.DataFrame
    ----------
    Return:
    df_cleaned: pandas.DataFrame
    """

    def __init__(self, flag_dict):
        """Class initilization
        Parameters
        ----------
        flag_dict: dict of regex to replace by flag

        """
        self.flag_dict = flag_dict

    def fit(self, df):
        return self

    def transform(self, df):
        df_cleaned = df.copy()
        df_cleaned[stg.AMENDEMENT] = df_cleaned[stg.AMENDEMENT].apply(lambda x: str(x))
        df_cleaned[stg.AMENDEMENT] = df_cleaned[stg.AMENDEMENT].apply(self.lemmatizer)
        df_cleaned[stg.AMENDEMENT] = df_cleaned[stg.AMENDEMENT].apply(self.lowercase)
        df_cleaned[stg.AMENDEMENT] = df_cleaned[stg.AMENDEMENT].apply(
            self.flag_text, args=(self.flag_dict,)
        )
        return df_cleaned

    @staticmethod
    def lowercase(text):
        return text.lower()

    @staticmethod
    def remove_accents(text, method="unicodedata"):

        if method == "unidecode":
            return unidecode.unidecode(text)
        elif method == "unicodedata":
            utf8_str = (
                unicodedata.normalize("NFKD", text)
                .encode("ASCII", "ignore")
                .decode("utf-8")
            )
            return utf8_str
        else:
            raise ValueError(
                "Possible values for method are 'unicodedata' or 'unidecode'"
            )

    @staticmethod
    def flag_text(text, flag_dict):
        for regex, flag in flag_dict.items():
            text = re.sub(regex, flag, str(text))

        return text

    @staticmethod
    def lemmatizer(text):
        nlp = fr_core_news_md.load(disable=["ner", "parser"])
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        doc = nlp(text)
        text_lemm = [token.lemma_ for token in doc]
        text_lemm = " ".join(text_lemm)
        return str(text_lemm)


class TopicWordsFinder:
    """
    Get amendements df with labels from model and perform clustering words recognition

    Attributes
    ----------
    df: pandas.DataFrame
    ----------
    Return:

    words_per_topic: Array of top words with tf-idf score per topic
    topic_size: pandas.DataFrame
    """

    def __init__(self, df):
        """Class initilization
        Parameters
        ----------
        df: cleaned and lemmatized dataframe with topic already assigned from previous clusterization
        """

        self.df = df

    @property
    def words_per_topic(self):
        df_topic = self.get_df_topic(self.df)
        count_vectorizer = self.get_countvectorizer(df_topic[stg.AMENDEMENT])
        c_tf_idf = self.get_custom_tfidf(
            df_topic[stg.AMENDEMENT], len(self.df), count_vectorizer
        )
        top_n_words = self.extract_top_n_words_per_topic(
            c_tf_idf, count_vectorizer, df_topic, n=20
        )
        topic_sizes = self.extract_topic_sizes(self.df)
        return top_n_words, topic_sizes

    def get_df_topic(self, df):
        df_topic = df.copy()
        df_topic[stg.AMENDEMENT] = df_topic[stg.AMENDEMENT].apply(lambda x: str(x))
        df_topic = (
            df_topic.groupby([stg.TOPIC]).agg({stg.AMENDEMENT: " ".join}).reset_index()
        )
        return df_topic

    def get_countvectorizer(self, doc):
        count_vectorizer = CountVectorizer(**stg.PARAMS_CV)
        count_vectorizer.fit(doc)
        return count_vectorizer

    def get_custom_tfidf(self, doc, m, count_vectorizer):
        topics = count_vectorizer.transform(doc).toarray()
        words = topics.sum(axis=1)
        tf = np.divide(topics.T, words)
        sum_t = topics.sum(axis=0)
        idf = np.log(np.divide(1 + m, 1 + sum_t) + 1).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)
        return tf_idf

    def extract_top_n_words_per_topic(self, tf_idf, count_vectorizer, df_topic, n):
        words = count_vectorizer.get_feature_names()
        labels = list(df_topic[stg.TOPIC])
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {
            label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
            for i, label in enumerate(labels)
        }
        return top_n_words

    def extract_topic_sizes(self, df):
        topic_sizes = (
            df.groupby([stg.TOPIC], sort=True)[stg.AMENDEMENT]
            .count()
            .reset_index()
            .rename({stg.AMENDEMENT: "Size"}, axis="columns")
            .sort_values("Size", ascending=False)
        )
        return topic_sizes


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(stg.DATA_DIR, "amendements_sentence.csv"))
    df = df.iloc[::25000]
    cleaner = TextCleaner(flag_dict=stg.FLAG_DICT)
    df_cleaned = cleaner.transform(df)
    print(df_cleaned[stg.AMENDEMENT].head())
    df_f = df_cleaned.copy()
    df_f["Topic"] = 0
    import random

    for i in range(df_f.shape[0]):
        df_f["Topic"].iloc[i] = random.randrange(0, 3, 1)
    word_finder = TopicWordsFinder(df_f)
    top_n_words, topic_sizes = word_finder.words_per_topic
    for i in range(0, (len(topic_sizes))):
        print(i, top_n_words[i][:10])
    print(topic_sizes.head(25))
