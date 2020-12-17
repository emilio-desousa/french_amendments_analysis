import amendements_analysis.settings.base as stg
from amendements_analysis.infrastructure.building_dataset import DatasetBuilder
from amendements_analysis.infrastructure.sentence_encoding import DatasetCleaner, TextEncoder
from amendements_analysis.domain.lda_model import LatentDirichletAllocationModel
from amendements_analysis.domain.topic_predicter import topic_predicter
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from streamlit.components.v1 import html
import umap.umap_ as umap
import os
import pickle
import numpy as np


#sentence_embeddings = np.load(os.path.join(stg.DATA_DIR, "FinedTunedBert_fullDF.npy"))

#umap_model = umap.UMAP(n_neighbors=15, 
#                        n_components=15, 
#                        metric='cosine')
#pickle.dump(umap_model_fit, open(os.path.join(stg.DATA_DIR,'umap_model_fitted.sav'), 'wb'))

a = ["""Cet amendement du groupe Socialistes et apparentés vise à fermer complètement la porte à la minoration des variables d'ajustement , à hauteur de 50 millions d’euros, de deux concours financiers de l’État aux collectivités territoriales.

Pour rappel, l’article 22  dans sa rédaction initiale minore de 50 millions deux concours financiers de l'Etat aux collectivités territoriales (appelés variables d’ajustement), à savoir :

Dotation de compensation de la réforme de la taxe professionnelle (DCRTP) :

- Part régionale : -7,5 millions d'euros
- Part départementale : -5 millions d'euros

Dotation de transfert des compensations d’exonération de taxe d’habitation (DTCE) :

- Part régionale : -17,5 millions d'euros
- Part départementale : -20 millions d'euros

Ces minorations avaient atteint 120 millions d’euros en 2020, 159 millions d’euros en 2019 et 293 millions d’euros en 2018.

La DCRTP et la DTCE ont été créées lors de la réforme de la taxe professionnelle et de la fiscalité locale, intervenue en 2010, pour compenser les collectivités perdantes dans le cadre de cette réforme, en vue d’en assurer la neutralité financière. Ces dotations, qui se substituaient à des ressources fiscales dynamiques, avaient donc vocation à être figées sur le montant initialement fixé.

L’introduction de la DCRTP et de la DTCE au sein des variables d’ajustement est donc une mesure injuste et difficilement acceptable pour les régions et départements."""]


sentence = TextEncoder(a,
finetuned_bert = False, batch_size=1).sentence_embeddings

umap_model_fit = pickle.load(open(os.path.join(stg.DATA_DIR,'umap_model_fitted.sav'), 'rb'))
cluster_model_fit = pickle.load(open(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME),"rb"))
prediction = topic_predicter(sentence, umap_model_fit, cluster_model_fit).predicted_topic
print('prediction is : ', prediction)