import amendements_analysis.settings.base as stg
from amendements_analysis.infrastructure.building_dataset import DatasetBuilder
from amendements_analysis.infrastructure.sentence_encoding import (
    DatasetCleaner,
    TextEncoder,
)
from amendements_analysis.domain.lda_model import LatentDirichletAllocationModel
from amendements_analysis.domain.topic_predicter import topic_predicter
import streamlit as st
import matplotlib.pyplot as plt
from streamlit.components.v1 import html
import umap.umap_ as umap
import pickle
import joblib
import os
import numpy as np


# Selection of the method of classification
method_of_classification = st.sidebar.selectbox(
    "Sélectionner la méthode de classification", ("UMAP", "LDA")
)

# Parameters settings
if method_of_classification == "UMAP":
    # Loading the UMAP model
    st.markdown("**Classification UMAP**")
    cluster_model_fit = pickle.load(
        open(os.path.join(stg.DATA_DIR, stg.CLUSTER_MODEL_FIT_FILENAME), "rb")
    )
    umap_model_fit = joblib.load(
        open(os.path.join(stg.DATA_DIR, stg.DATA_DIR, stg.UMAP_MODEL_FIT_FILENAME_EXT), "rb")
    )
    amendement = st.text_input("Amendement à classifier")
    submit = st.button("Prediction")
    if submit:
        amendement = [amendement]
        sentence_embeddings = TextEncoder(
            amendement, finetuned_bert=True, batch_size=1
        ).sentence_embeddings
        prediction = topic_predicter(
            sentence_embeddings, umap_model_fit, cluster_model_fit
        ).predicted_topic
        st.write("Le sujet identifié est : ", prediction)
elif method_of_classification == "LDA":
    st.markdown("**Latent dirichlet allocation**")
    # Loading of the LDA model
    with open(os.path.join(stg.DATA_DIR, stg.LDA_MODEL_VIS_FILENAME), "r") as f:
        html_string = f.read()
        html(html_string, width=1300, height=800)
