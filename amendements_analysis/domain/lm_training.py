import os
import amendements_analysis.settings.base as stg
from amendements_analysis.infrastructure.building_dataset import DatasetBuilder
from datasets import load_dataset
from transformers import (
    CamembertForMaskedLM,
    CamembertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class LM_Trainer:
    def __init__(self):
        self.tokenizer = CamembertTokenizerFast.from_pretrained(
            "camembert-base", max_len=512
        )
        self.model = CamembertForMaskedLM.from_pretrained("camembert-base")
        self.training_args = TrainingArguments(
            output_dir=stg.RES_DIR,
            overwrite_output_dir=True,
            # num_train_epochs=3,
            # per_device_train_batch_size=16,
            # fp16=True,
            save_steps=10_000,
            save_total_limit=2,
        )

    def _create_datasets(self):
        builder = DatasetBuilder()
        builder.create_train_test_files(builder.data, force_write_csv=False)
        data_files = {}
        data_files["train"] = os.path.join(stg.INTERIM_DIR, stg.CSV_DATA_TRAIN_LM)
        data_files["validation"] = os.path.join(stg.INTERIM_DIR, stg.CSV_DATA_TEST_LM)
        datasets = load_dataset(stg.DATASET_FILE_FORMAT, data_files=data_files)
        return (
            datasets,
            datasets["train"].column_names,
        )

    def _tokenize_function(self, row):
        row[stg.AMENDEMENT] = [
            line for line in row[stg.AMENDEMENT] if len(line) > 0 and not line.isspace()
        ]
        return self.tokenizer(
            row[stg.AMENDEMENT],
            padding=True,
            truncation=True,
            max_length=512,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    def train(self):
        datasets, column_names = self._create_datasets()
        tokenized_datasets = datasets.map(self._tokenize_function, batched=True)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        lm_trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        lm_trainer.train()


if __name__ == "__main__":
    test = LM_Trainer()
    test.train()
