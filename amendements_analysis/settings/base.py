import os

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
test = "bla"

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
FLAG_DICT = {
    r'/^(?!(?:1975)$)\d+/' : 'flag_numero',
    r'\bANCT\b' : 'flag_territoire',
    r'\b1975\b' : 'flag_retraite',
    r'\b[A-Za-z]\b': ''
}
TOPICS_DICT = {
    '0': 'Administratif (vote, scrutin, marchés publics)',
    '1': 'Retrait & contestation de loi / decret etc',
    '2': 'Immobilier & urbanisme',
    '3': 'Naissances & procreation',
    '4': 'Justice, sécurité, pénal',
    '5': 'Politique echelle commune & regions',
    '6': 'Travail,Entreprises',
    '7': 'Fiscalité des particuliers',
    '8': 'Santé, Handicap',
    '9': 'Ecologie, Traitement dechet'
}


STOPWORDS_TO_ADD = ['amendement', 'article', 'cette', 'cet', 'cela', 'leurs', 'plus', 'afin', 'donc', 'ores', 'etre']
