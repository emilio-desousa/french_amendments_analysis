import json


def load_data(path):
    """
    Opens the json file at path location (from current dir)
    Parameters
    ----------
    path: str
        Path to json file
    Returns
    -------
    json_data: {}
        Data
    """
    with open(path, "r") as f:
        json_data = json.load(f)
    return json_data


with open("../../../data/DLR5L14N34301/PIONANR5L15B0422/AMANR5L15PO59051B0422P0D1N000002.json") as f:
    json_data = json.load(f)
import unicode

text = unicode(json_data["amendement"]["corps"], "utf-8")
test = json_data[u"amendement"][u"corps"]
print(test)
test2 = "é"
