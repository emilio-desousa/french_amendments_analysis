import amendements_analysis.settings.base as stg
import umap.umap_ as umap
import numpy as np
from sklearn_extra.cluster import KMedoids
import os


class clusters_finder:
    """
    Get sentences embeddings and generate cluster according to the number of cluster previously defined.
    An UMAP dimension reduction and a Kmenoid with cosine distance are performed for this task.

    Attributes:
        sentence_embeddings (numpy array) : sentence embeddings

    """

    def __init__(self, sentence_embeddings, n_umap=15, n_clusters=13):
        """Class initilization
        Parameters:
            sentence_embedding numpy.Array : numpy array of previous Bert embedding
            n_clusters list : number of topics to split the dataset
            n_umap  list : number of axis after UMAP size reduction
        """
        self.sentence_embeddings = sentence_embeddings
        self.n_clusters = n_clusters
        self.n_umap = n_umap

    @property
    def models(self):
        """
         Get sentences embeddings and generate cluster according to the number of cluster previously defined.
        An UMAP dimension reduction and a Kmenoid with cosine distance are performed for this task.

        Returns:
            sklearn.model:  fitted umap model with attribute 'sentence_embeddings'
        Returns:
            sklearn.model:  fitted kmenoid model from umaped 'sentence_embeddings'
        """
        umap_model = umap.UMAP(
            n_neighbors=15, n_components=self.n_umap, metric="cosine"
        )
        umap_model = umap_model.fit(self.sentence_embeddings)
        umap_embeddings = umap_model.transform(self.sentence_embeddings)
        kmenoid_model = KMedoids(
            n_clusters=self.n_clusters, metric="cosine", init="random", random_state=15
        )
        cluster = kmenoid_model.fit(umap_embeddings)
        return cluster, umap_model


if "__main__" == __name__:
    sentence_embeddings = np.load(
        os.path.join(stg.DATA_DIR, stg.SENTENCE_EMBEDDINGS_FILENAME)
    )
    sentence_embeddings = sentence_embeddings[::300]
    cluster, umap_model = clusters_finder(sentence_embeddings).models
    print("labels are : ", cluster.labels_)
    print("clusters centers are : ", cluster.cluster_centers_)
