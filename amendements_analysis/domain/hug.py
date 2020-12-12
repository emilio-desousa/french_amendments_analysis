import amendements_analysis.settings.base as stg
import torch
import ast
import os
from pathlib import Path
import logging
import pandas as pd
from transformers import BertTokenizerFast

from sklearn.model_selection import train_test_split

path = os.path.join(stg.DATA_DIR, "amendements_sentence.csv")
df = pd.read_csv(path)
df.drop_duplicates(subset="expose_sommaire", keep=False, inplace=True)
train_dataset = df.iloc[::3, :]["expose_sommaire"]


class AmendementDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


(
    X_train,
    X_test,
) = train_test_split(train_dataset, test_size=0.33, random_state=42)
tokenizer = BertTokenizerFast.from_pretrained("camembert-base")

train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

train_dataset = AmendementDataset(train_encodings)
test_dataset = AmendementDataset(test_encodings)

from transformers import BertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=stg.RES_DIR,  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=stg.LOGS_DIR,  # directory for storing logs
    logging_steps=10,
)
model = BertForSequenceClassification.from_pretrained("camembert-base")

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
)

trainer.train()