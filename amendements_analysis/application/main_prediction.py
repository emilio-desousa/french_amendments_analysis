import amendements_analysis.settings.base as stg
from amendements_analysis.domain.topic_predicter import topic_predicter
from amendements_analysis.infrastructure.sentence_encoding import TextEncoder
import pickle
import joblib
import os

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

if __name__ == "__main__" :
    main_model_predict()