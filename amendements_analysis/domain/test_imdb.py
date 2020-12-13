from pathlib import Path
import os
import amendements_analysis.settings.base as stg
import pandas as pd


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

train_texts, train_labels = train_texts[:2000], train_labels[:2000]
test_texts, test_labels = test_texts[:200], test_labels[:200]

train_file = pd.read_csv(os.path.join(stg.INTERIM_DIR, "train_dataset_lm.csv"))
train_data = list(train_file["expose_sommaire"])
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding="max_length", max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding="max_length", max_length=512)


import torch


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

from transformers import BertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=stg.RES_DIR,  # output directory
    num_train_epochs=1,  # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=4,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=stg.LOGS_DIR,  # directory for storing logs
    logging_steps=10,
)

model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
)

trainer.train()