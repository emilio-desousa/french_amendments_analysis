import amendements_analysis.settings.base as stg

from amendements_analysis.infrastructure.building_dataset import DatasetBuilder

from amendements_analysis.infrastructure.sentence_encoding import DatasetCleaner
from amendements_analysis.infrastructure.sentence_encoding import TextEncoder

from amendements_analysis.domain.clusters_finder import clusters_finder
from amendements_analysis.domain.topic_finder import TextCleaner, TopicWordsFinder

from amendements_analysis.domain.topic_predicter import topic_predicter

import pickle
import joblib
import os, os.path
import pandas as pd


def main_model_train(clusters=13, partition=2, lemma=True, do_cluster=True):
    print("=============Preparing download of data...==============")

    builder = DatasetBuilder(is_split_sentence=False, rewrite_csv=False, get_only_amendments=False)
    df = builder.data
    print("===========Dataset is loaded and ready to be used================")
    print("===========Dataset will be now encoded by camemBERT model================")
    cleaner = DatasetCleaner(df, partition)
    sentences = cleaner.sentences
    # df_bert = cleaner.df_bert
    encoder = TextEncoder(sentences, finetuned_bert=True, batch_size=2)
    sentence_embeddings = encoder.sentence_embeddings
    print("===========Sentences of the dataset have been encoded according to camemBERT================")
    if do_cluster == True:
        cluster_model_fit, umap_model_fit = clusters_finder(sentence_embeddings, n_clusters=clusters).models
        joblib.dump(umap_model_fit, os.path.join(stg.DATA_DIR, stg.UMAP_MODEL_FIT_FILENAME_EXT), compress=0)
        pickle.dump(cluster_model_fit, open(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME_EXT), "wb"))
    else:
        if os.path.isfile(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME_EXT)) and os.path.isfile(
            os.path.join(stg.DATA_DIR, stg.UMAP_MODEL_FIT_FILENAME_EXT)
        ):
            cluster_model_fit = pickle.load(
                open(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME_EXT), "rb")
            )
            umap_model_fit = joblib.load(os.path.join(stg.DATA_DIR, stg.UMAP_MODEL_FIT_FILENAME_EXT))

    print("===========Cluster & UMAP fit models from camemBERT embedding available================")
    df_cluster = pd.read_csv(os.path.join(stg.DATA_DIR, "lemmatized.csv"))
    cleaner_lemmatizer = TextCleaner(flag_dict=stg.FLAG_DICT)
    if lemma == True:
        cleaner_lemmatizer = TextCleaner(flag_dict=stg.FLAG_DICT)
        df_cleaned = cleaner_lemmatizer.transform(df_cluster)
        df_cleaned.to_csv(os.path.join(stg.DATA_DIR, stg.DF_CLEANED_LEMMA_PATH), index=False)
    else:
        df_cleaned = pd.read_csv(os.path.join(stg.DATA_DIR, stg.DF_CLEANED_LEMMA_PATH))

    df_cleaned[stg.TOPIC] = cluster_model_fit.labels_
    print("===========dataframe used for cluster is cleaned & lemmatized to find important words================")

    word_finder = TopicWordsFinder(df_cleaned)
    top_n_words, topic_sizes = word_finder.words_per_topic
    for i in range(0, (len(topic_sizes))):
        print(i, top_n_words[i][:10])
    print(topic_sizes.head(13))
    print("===========From topic top words, decide to attribute a topic for each topic number================")
    topic_dict = {}
    for i in range(topic_sizes.shape[0]):
        topic = input("topic number {}:".format(i))
        topic_dict[str(i)] = topic
    print("===========Models are pickled or Joblibed================")
