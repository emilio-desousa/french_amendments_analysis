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

def main_model_train():
    print('=============Preparing download of data...==============')

    builder = DatasetBuilder(is_split_sentence=False, rewrite_csv=False, get_only_amendments=False)
    df = builder.data
    print('===========Dataset is loaded and ready to be used================')
    print('===========Dataset will be now encoded by camemBERT model================')
    cleaner = DatasetCleaner(df, partition = 5000)
    sentences = cleaner.sentences
    df_bert = cleaner.df_bert
    encoder = TextEncoder(sentences, finetuned_bert = True, batch_size=2)
    sentence_embeddings = encoder.sentence_embeddings
    print('===========Sentences of the dataset have been encoded according to camemBERT================')

    cluster, umap_model = clusters_finder(sentence_embeddings, n_clusters = 3).models
    print('===========Cluster & UMAP fit models from camemBERT embedding available================')

    cleaner_lemmatizer = TextCleaner(flag_dict=stg.FLAG_DICT)
    df_cleaned = cleaner_lemmatizer.transform(df_bert)
    df_cleaned[stg.TOPIC] = cluster.labels_
    print(df_cleaned['Topic'])
    print('===========dataframe used for cluster is cleaned & lemmatized to find important words================')

    word_finder = TopicWordsFinder(df_cleaned)
    top_n_words, topic_sizes = word_finder.words_per_topic
    for i in range(0,(len(topic_sizes))):
        print(i, top_n_words[i][:10])
    print(topic_sizes.head(13))
    print('===========From topic top words, decide to attribute a topic for each topic number================')

    joblib.dump(umap_model, os.path.join(stg.DATA_DIR, stg.UMAP_MODEL_FIT_FILENAME_EXT))
    pickle.dump(cluster, open(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME_EXT), 'wb'))

    print('===========Models are pickled or Joblibed================')

def main_model_predict():

    amendement_list = []
    amendement = input("Enter one law text: ")
    amendement_list.append(amendement)
    encoder_predict = TextEncoder(amendement_list, finetuned_bert = True, batch_size=1)
    sentence_embedding = encoder_predict.sentence_embeddings
    try:
        prediction_number, _ = topic_predicter(sentence_embedding, umap_model, cluster).predicted_topic
        print('prediction is : ', prediction_number)
    except NameError:
        if os.path.isfile(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME_EXT)) and\
           os.path.isfile(os.path.join(stg.DATA_DIR, stg.UMAP_MODEL_FIT_FILENAME_EXT)):
            cluster_model_fit = pickle.load(open(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME_EXT), "rb"))
            umap_model_fit = joblib.load(os.path.join(stg.DATA_DIR, stg.UMAP_MODEL_FIT_FILENAME_EXT))
            prediction_number, _ = topic_predicter(sentence_embedding, umap_model_fit, cluster_model_fit).predicted_topic
            print('prediction (from saved model) is : ', prediction_number)
        else:
            print('No model UMAP & CLUSTERS are available, please generate one')

if __name__ == "__main__":
    main_model_train()
    main_model_predict()
