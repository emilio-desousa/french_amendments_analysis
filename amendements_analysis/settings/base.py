import os

RANDOM_STATE = 42


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(REPO_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
JSON_DIR = os.path.join(RAW_DATA_DIR, "json/")

RES_DIR = os.path.join(REPO_DIR, "results/")
MODEL_DIR = os.path.join(RES_DIR, "models/")
TMP_MODEL_DIR = os.path.join(MODEL_DIR, "tmp_model/")
CUSTOM_MODEL_REPO_DIR = os.path.join(MODEL_DIR, "camembert_aux_amandes/")
MODEL_REPO_URL = (
    "https://fenrhjen:CamembertPublic01_@huggingface.co/fenrhjen/camembert_aux_amandes/"
)
DF_CLEANED_LEMMA_PATH = 'df_amendement_cleaned_lemma.csv'
NEEDED_FILES_MODEL = [
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
]
CSV_DATA_TRAIN_LM = "train_dataset_lm.csv"
CSV_DATA_TEST_LM = "test_dataset_lm.csv"
CSV_DATA_SPLITTED_FILENAME = "amendements_splitted.csv"
ZIP_FILE_NAME = "Amendements_XV.json.zip"
URL_TO_DL_DATA = "http://data.assemblee-nationale.fr/static/openData/repository/15/loi/amendements_legis/Amendements_XV.json.zip"
REGULAR_CAMEMBERT = "camembert-base"
FINED_TUNED_CAMEMBERT = "fenrhjen/camembert_aux_amandes"

SENTENCE_EMBEDDINGS_FILENAME = "FinedTunedBert_fullDF.npy"
UMAP_EMBEDDINGS_FILENAME = "umap_15_axis_BertFT_half.npy"
CLUSTER_MODEL_FIT_FILENAME = "cluster_model_fit"
UMAP_MODEL_FIT_FILENAME = "umap_model_fit"
CLUSTER_MODEL_FIT_FILENAME_EXT = "cluster_model_fit.sav"
UMAP_MODEL_FIT_FILENAME_EXT = "umap_model_fit.sav"
LDA_MODEL_VIS_FILENAME = "pyLDAvis.html"

SOURCE_COLUMNS = [
    "amendement",
    "uid",
    "texteLegislatifRef",
    "corps",
]


def get_stopwords():
    stopwords = [
        line.rstrip("\n")
        for line in open(os.path.join(os.path.dirname(__file__), "stopwords.txt"))
    ]
    return stopwords


COLUMNS = [
    "uid",
    "texte_legislatif_ref",
    "dispositif",
    "expose_sommaire",
    "date_depot",
    "date_publication",
    "date_sort",
    "sort",
    "etat",
    "sous_etat",
]

AMENDEMENT = "expose_sommaire"
TOPIC = "Topic"

DATASET_FILE_FORMAT = "csv"
PARAMS_CV = {
    "strip_accents": "unicode",
    "stop_words": get_stopwords(),
    "max_df": 0.9,
    "min_df": 0.01,
    "ngram_range": (1, 2),
}

FLAG_DICT = {
    r"\b1975\b": "flag_retraite",
    r"\d+": "flag_numero",
    r"\bANCT\b": "flag_territoire",
    r"\b[A-Za-z]\b": "",
    r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})": "flag_url",
}

TOPICS_DICT = {
    "0": "Ecologie, Rechauffement climatique",
    "1": "Justice, Droit, Sécurité, Pénal",
    "2": "Retraites",
    "3": "Naissances, Procreation & condition de la Jeunesse",
    "4": "Medecine & Santé",
    "5": "Fiscalité / Aides des Entreprises & Particuliers",
    "6": "Politique Cohésion Territoriale (National)",
    "7": "Immobilier & Urbanisme",
    "8": "Alimentaire, Elevage, Agriculture & traitement des Déchets",
    "9": "Rectification / Contestation de lois (vote, scrutin)",
    "10": "Service Publique (éducation, transports publics, radio ...)",
    "11": "Financements des Secteurs",
    "12": "Gestion Budgétaire",
}
TOPICS_DICT_CUSTOM = {}

STOPWORDS_TO_ADD = [
    "amendement",
    "article",
    "cette",
    "cet",
    "cela",
    "leurs",
    "plus",
    "afin",
    "donc",
    "ores",
    "etre",
    "nous",
    "socialistes",
]

PARAMETERS_CV = {
    "strip_accents": "unicode",
    "lowercase": True,
    "min_df": 0.01,
    "max_df": 0.50,
    "ngram_range": (1, 2),
    "token_pattern": r"\b[^\d\W]+\b",
}

PARAMETERS_LDA = {
    "n_components": 10,
    "doc_topic_prior": 0.7,
    "topic_word_prior": 0.5,
    "verbose": 1,
}
