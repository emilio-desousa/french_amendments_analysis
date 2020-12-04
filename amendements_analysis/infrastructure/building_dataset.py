import amendements_analysis.settings.base as stg
import os
import re
import glob
import html
import pandas as pd
import json
import zipfile
import wget
from tqdm import tqdm


class DatasetBuilder:
    """
    Get amendments from the Web if csv file or zip file doesn't exist

    Attributes
    ----------
    is_split_sentence: bool
    rewrite_csv: bool

    Properties
    ----------
    data: pandas.DataFrame
    """

    def __init__(self, is_split_sentence=True, rewrite_csv=False):
        """Class initilization

        Parameters
        ----------
        is_split_sentence : bool, optional
            set True if you want the amendment splitted in sentences, by default True
        rewrite_csv : bool, optional
            Set True if you want to overwrite csv from json files, by default False
        """
        self.is_split_sentence = is_split_sentence
        self.rewrite_csv = rewrite_csv

    @staticmethod
    def cleanString(text):
        """Clean a string by unescape html char and tags

        Parameters
        ----------
        text : str
            str to clean

        Returns
        -------
        str
            Cleaned str
        """
        text_unescaped = html.unescape(text)
        p = re.compile(r"<.*?>")
        return p.sub("", text_unescaped)

    @staticmethod
    def split_text_into_sentences(data):
        """Method to split a text into sentences

        Parameters
        ----------
        data : str
            text to split

        Returns
        -------
        list
            list of sentences got by the split
        """
        reg = re.compile(r"(?<!\b[M]|\b[Dr])[.?!]\s*(?=[A-Z])")
        return reg.split(data)

    @property
    def data(self):
        """Property to get the Dataframe,
        If csv exists => use it
        unless If Zip exists => use it
        unless download Data from Web

        Returns
        -------
        pandas.DataFrame
            dataframe witl all amendments
        """
        if not os.path.isfile(os.path.join(stg.RAW_DATA_DIR, stg.ZIP_FILE_NAME)):
            self._download_data(os.path.join(stg.RAW_DATA_DIR, stg.ZIP_FILE_NAME))

        if os.path.isfile(os.path.join(stg.INTERIM_DIR, stg.CSV_DATA_SPLITTED_FILENAME)) and not self.rewrite_csv:
            df = pd.read_csv(os.path.join(stg.INTERIM_DIR, stg.CSV_DATA_SPLITTED_FILENAME))
        else:
            df = self._get_data_from_all_jsons()
        return df

    def _download_data(self, zip_file):
        print("Downloading ZipFile... ")
        wget.download(stg.URL_TO_DL_DATA, zip_file)
        print("Done!")
        print("Extracting...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(stg.RAW_DATA_DIR)
        print("Done!")

    def _get_data_from_all_jsons(self):
        files = glob.glob(os.path.join(stg.JSON_DIR, "**/**/*.json"))
        amendements = []
        index = 0
        for file in tqdm(files, total=len(files)):
            with open(file) as f:
                index = index + 1
                json_file = json.load(f)
                amendements.append(self._get_amendement_from_json(json_file, self.is_split_sentence))
        df = pd.DataFrame(data=amendements, columns=stg.COLUMNS)
        df.to_csv(os.path.join(stg.INTERIM_DIR, stg.CSV_DATA_SPLITTED_FILENAME), index=False)
        return df

    def _get_amendement_from_json(self, json_file, is_split_sentence=True):
        amendement_array = []
        amendement = json_file["amendement"]
        amendement_array.append(amendement["uid"])
        amendement_array.append(amendement["texteLegislatifRef"])
        corps = amendement["corps"]

        if "dispositif" in corps["contenuAuteur"]:
            dispositif_json = self.cleanString(corps["contenuAuteur"]["dispositif"])
            dispositif = self.split_text_into_sentences(dispositif_json) if is_split_sentence else dispositif_json
            amendement_array.append(dispositif)
        else:
            amendement_array.append(None)
        if "exposeSommaire" in corps["contenuAuteur"]:
            expose_sommaire_json = self.cleanString(corps["contenuAuteur"]["exposeSommaire"])
            expose_sommaire = (
                self.split_text_into_sentences(expose_sommaire_json) if is_split_sentence else expose_sommaire_json
            )
            amendement_array.append(expose_sommaire)
        else:
            amendement_array.append(None)
        cycle_de_vie = amendement["cycleDeVie"]
        amendement_array.append(cycle_de_vie["dateDepot"])
        amendement_array.append(cycle_de_vie["datePublication"])
        amendement_array.append(cycle_de_vie["dateSort"])
        amendement_array.append(cycle_de_vie["sort"])
        if "libelle" in cycle_de_vie["etatDesTraitements"]["etat"]:
            amendement_array.append(cycle_de_vie["etatDesTraitements"]["etat"]["libelle"])
        else:
            amendement_array.append(None)
        if "libelle" in cycle_de_vie["etatDesTraitements"]["sousEtat"]:
            amendement_array.append(cycle_de_vie["etatDesTraitements"]["sousEtat"]["libelle"])
        else:
            amendement_array.append(None)
        return amendement_array


if __name__ == "__main__":
    test = DatasetBuilder(rewrite_csv=False)
    test.data
