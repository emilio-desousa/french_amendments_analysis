import torch
import amendements_analysis.settings.base as stg
from amendements_analysis.infrastructure.building_dataset import DatasetBuilder
import ast
import os
from pathlib import Path
import logging

logger = logging.getLogger()

data_builder = DatasetBuilder(is_split_sentence=False, is_only_summary=True, rewrite_csv=False)
df = data_builder.data

dataset = set(df["summary"][1:10])
del df
device = torch.device("cuda")
data_dir = os.path.join(stg.INTERIM_DIR, stg.CSV_DATA_WITH_ONLY_SUMMARY)


from fast_bert.data_lm import BertLMDataBunch

databunch = BertLMDataBunch.from_raw_corpus(
    data_dir=Path(stg.INTERIM_DIR),
    text_list=list(dataset),
    tokenizer="camembert-base",
    batch_size_per_gpu=1,
    max_seq_length=256,
    model_type="camembert-base",
    logger=logger,
)


from fast_bert.learner_lm import BertLMLearner

learner = BertLMLearner.from_pretrained_model(
    dataBunch=databunch,
    pretrained_path="camembert-base",
    output_dir=Path(stg.INTERIM_DIR),
    metrics=[],
    device=device,
    logger=logger,
    is_fp16=True,
    multi_gpu=False,
    logging_steps=50,
)
learner.fit(
    epochs=2,
    lr=0.01,
    # validate=True, 	# Evaluate the model after each epoch
    schedule_type="warmup_cosine_hard_restarts",
)
