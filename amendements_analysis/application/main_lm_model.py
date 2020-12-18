import amendements_analysis.settings.base as stg
from amendements_analysis.settings.utils import update_progress
from amendements_analysis.domain.lm_training import LM_Trainer
from amendements_analysis.domain.tranformer_uploader import Model_Publisher

import pandas as pd
from os.path import join
import argparse


def main():
    PARSER = argparse.ArgumentParser(description="Language Model Trainer")

    PARSER.add_argument(
        "--publish",
        "-p",
        required=False,
        help="True or false => Publish model to huggingface, you need the credentials",
    )

    ARGS = PARSER.parse_args()

    PUBLISH = f"{ARGS.publish}"
    # update_progress(0)
    print("progress : ")
    trainer = LM_Trainer()
    trainer.train()
    if PUBLISH == True:
        Model_Publisher().push_model()
        print("You can now find your model on huggingFace!")
    else:
        print("Done")


if __name__ == "__main__":
    main()
