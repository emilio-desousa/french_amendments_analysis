import amendements_analysis.settings.base as stg
import numpy as np
from sklearn_extra.cluster import KMedoids
import umap.umap_ as umap
import os
import pickle
import joblib


class topic_predicter:
    """
    Get amendement and attribute a topic.
    ------------
    Attributes
    ----------
    df: pandas DataFrame with
    umap_model = model to reduce dimension
    kmenoid_model_fitted: model
    ----------
    Return:
    predicted_topic : string with topic prediction
    """

    def __init__(self, sentence_embedding, umap_model_fit, cluster_model_fit):
        """Class initilization
        Parameters
        ----------
        sentence_embedding: numpy array of previous Bert embedding
        umap_model_fit = fit model for umap dimension reduction
        cluster_model_fit = fit model for clustering prediction
        """
        self.sentence_embedding = sentence_embedding
        self.cluster_model_fit = cluster_model_fit
        self.umap_model_fit = umap_model_fit

    @property
    def predicted_topic(self):
        # umap_embedding = self.umap_model_fit.transform(self.sentence_embedding.reshape(1, -1))
        # print(self.sentence_embedding)

        print(self.sentence_embedding)
        umap_embedding = self.umap_model_fit.transform(
            self.sentence_embedding.reshape(1, -1)
        )
        predict = self.cluster_model_fit.predict(umap_embedding)
        prediction_number = str(predict[0])
        prediction_name = stg.TOPICS_DICT[str(predict[0])]
        return prediction_number, prediction_name


if "__main__" == __name__:
    sentence_embeddings = np.load(
        os.path.join(stg.DATA_DIR, stg.SENTENCE_EMBEDDINGS_FILENAME)
    )
    umap_model = umap.UMAP(n_neighbors=15, n_components=15, metric="cosine")
    umap_model_fit = umap_model.fit(sentence_embeddings[::100])
    cluster_model_fit = pickle.load(
        open(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME), "rb")
    )
    prediction = topic_predicter(
        sentence_embeddings[78563], umap_model_fit, cluster_model_fit
    ).predicted_topic
    print("prediction is : ", prediction)
