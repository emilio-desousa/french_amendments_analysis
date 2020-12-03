import amendements_analysis.settings.base as stg
import os
import re
import glob
import html
import pandas as pd
import json
import tqdm


class DatasetBuilder:
    def __init__(self, is_split_sentence):
        self.is_split_sentence = is_split_sentence

    @staticmethod
    def cleanString(data):
        data_unescaped = html.unescape(data)
        p = re.compile(r"<.*?>")
        return p.sub("", data_unescaped)

    @staticmethod
    def split_text_into_sentences(data):
        reg = re.compile(r"(?<!\b[M]|\b[Dr])[.?!]\s*(?=[A-Z])")
        return reg.split(data)

    @property
    def data(self):
        ##
        ##  [TODO] DEAL WITH EXISTED CSV
        ##
        print(os.path.join(stg.JSON_DIR, "**/**/*.json"))
        files = glob.glob(os.path.join(stg.JSON_DIR, "**/**/*.json"))
        amendements = []
        index = 0
        for file in files:
            with open(file) as f:
                index = index + 1
                json_file = json.load(f)
                if index == 100:
                    return pd.DataFrame(data=amendements, columns=stg.COLUMNS)
                amendements.append(self._get_amendement_from_json(json_file, self.is_split_sentence))
        return pd.DataFrame(data=amendements, columns=stg.COLUMNS)

    def _get_amendement_from_json(self, json_file, is_split_sentence=False):
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
    test = DatasetBuilder(is_split_sentence=False)
    test.data
