import amendements_analysis.settings.base as stg
from amendements_analysis.domain.topic_predicter import topic_predicter
from amendements_analysis.infrastructure.sentence_encoding import TextEncoder
import pickle
import joblib
import os

cluster_model = pickle.load(open(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME_EXT), "rb"))
umap_model = joblib.load(os.path.join(stg.DATA_DIR, stg.UMAP_MODEL_FIT_FILENAME_EXT))

amendement_list = []
amendement = input("Enter one law text: ")
amendement_list.append(amendement)

encoder_predict = TextEncoder(amendement_list, finetuned_bert = True, batch_size=1)
sentence_embedding = encoder_predict.sentence_embeddings
prediction_number, _ = topic_predicter(sentence_embedding, umap_model, cluster_model).predicted_topic
print('prediction is : ', prediction_number)
