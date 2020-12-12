from pathlib import Path
import os
import amendements_analysis.settings.base as stg


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels


train_texts, train_labels = read_imdb_split(os.path.join(stg.RAW_DATA_DIR, "aclImdb/train"))
test_texts, test_labels = read_imdb_split(os.path.join(stg.RAW_DATA_DIR, "aclImdb/test"))

from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("camembert-base")
