import os
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
STOPWORDS_LIST = stopwords.words("french")

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_DIR = os.path.join(REPO_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
JSON_DIR = os.path.join(RAW_DATA_DIR, "json/")

CSV_DATA_SPLITTED_FILENAME = "amendements_splitted.csv"
ZIP_FILE_NAME = "Amendements_XV.json.zip"
URL_TO_DL_DATA = (
    "http://data.assemblee-nationale.fr/static/openData/repository/15/loi/amendements_legis/Amendements_XV.json.zip"
)

SOURCE_COLUMNS = [
    "amendement",
    "uid",
    "texteLegislatifRef",
    "corps",
]

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
TOPIC = 'Topic'

PARAMS_CV = { 
    'strip_accents': 'unicode',
    'stop_words': STOPWORDS_LIST,
    'max_df': 0.9,
    'min_df': 0.01,
    'ngram_range': (1,2)
}

FLAG_DICT = {
    r'\b1975\b' : 'flag_retraite',
    r'\d+' : 'flag_numero',
    r'\bANCT\b' : 'flag_territoire',
    r'\b[A-Za-z]\b': '',
    r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})':'flag_url'
}

TOPICS_DICT = {
    '0': 'Ecologie, Rechauffement climatique',
    '1': 'Justice, Droit, Sécurité, Pénal',
    '2': 'Retraites',
    '3': 'Naissances, Procreation & condition de la Jeunesse',
    '4': 'Medecine & Santé',
    '5': 'Fiscalité / Aides des Entreprises & Particuliers' ,
    '6': 'Politique Cohésion Territoriale (National)',
    '7': 'Immobilier & Urbanisme',
    '8': 'Alimentaire, Elevage, Agriculture & traitement des Déchets',
    '9': 'Rectification / contestations de lois (vote, scrutin)',
    '10': 'Service Publique (éducations, transports publics ...)',
    '11': 'Financements des Secteurs',
    '12': 'Gestion Budgétaire'
}