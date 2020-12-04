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
