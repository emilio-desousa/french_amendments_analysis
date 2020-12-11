import amendements_analysis.settings.base as stg
from amendements_analysis.infrastructure.building_dataset import DatasetBuilder
from amendements_analysis.infrastructure.sentence_encoding import DatasetCleaner
from amendements_analysis.domain.lda_model import LatentDirichletAllocationModel
import pyLDAvis.sklearn
import pyLDAvis
import streamlit as st
import matplotlib.pyplot as plt
from streamlit.components.v1 import iframe
import pickle


# Selection of the method of classification
method_of_classification = st.sidebar.selectbox("Select classifier", ("KNN", 'LDA'))

# Parameters settings
if method_of_classification == "LDA":
    # Loading of the LDA model
    vis = pickle.load(open("../../data/viz.html", "rb"))
    params = {}
    topic_number = st.sidebar.slider("topic", 1, 10)
    params['topic'] = topic_number
    lam = st.sidebar.slider("lambda", 0., 1.)
    params["lambda"] = lam
else:
    params = {}


def get_classification(method_of_classification):
    if method_of_classification == "LDA":
        vis.topic_info["Lambda"] = round(1 - vis.topic_info["Freq"]/vis.topic_info["Total"], 1)
        st.write(vis.topic_info[(vis.topic_info.Category == 'Topic'+str(params['topic'])) & (vis.topic_info.Lambda < params['lambda'] + 0.1)]\
        .filter(["Term", "Freq"]))
    else:
        #fig = plt.figure()
        #plt.scatter(knn["x"], knn["y"], c=knn["topics"], cmap="viridis")
        #plt.colorbar()
        #st.pyplot(fig)
        pass


get_classification(method_of_classification)



