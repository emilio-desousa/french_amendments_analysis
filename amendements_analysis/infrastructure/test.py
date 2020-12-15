import os
import amendements_analysis.settings.base as stg
from datasets import load_dataset
from transformers import CamembertForMaskedLM
from transformers import CamembertTokenizerFast


tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base", max_len=512)
model = CamembertForMaskedLM.from_pretrained("camembert-base")

data_files = {}
data_files["train"] = os.path.join(stg.INTERIM_DIR, "train_dataset_lm.csv")
data_files["validation"] = os.path.join(stg.INTERIM_DIR, "test_dataset_lm.csv")
datasets = load_dataset("csv", data_files=data_files)


column_names = datasets["train"].column_names


def tokenize_function(examples):
    # Remove empty lines
    examples["expose_sommaire"] = [
        line for line in examples["expose_sommaire"] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples["expose_sommaire"],
        padding=True,
        truncation=True,
        max_length=512,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )


tokenized_datasets = datasets.map(tokenize_function, batched=True)
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
from transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
    output_dir=stg.RES_DIR,
    overwrite_output_dir=True,
    # num_train_epochs=3,
    # per_device_train_batch_size=16,
    # fp16=True,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)


# trainer.train()
