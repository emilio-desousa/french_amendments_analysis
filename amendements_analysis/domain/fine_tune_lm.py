from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs
import os


class LM_trainer:
    def __init__(self):
        self.train_file = os.path.join(os.getcwd(), "train_dataset.csv")
        self.test_file = os.path.join(os.getcwd(), "test_dataset.csv")

    def train(self):
        model_args = self._lm_args()
        model = LanguageModelingModel(
            "camembert",
            "camembert-base",
            use_cuda=False,
            args=model_args,
            train_files=self.train_file,
        )
        model.train_model(
            self.train_file,
            use_cuda=False,
            eval_file=self.test_file,
        )

    def _lm_args(self):
        model_args = LanguageModelingArgs()
        model_args.reprocess_input_data = True
        model_args.overwrite_output_dir = True
        model_args.num_train_epochs = 1
        model_args.learning_rate = 5e-4
        model_args.train_batch_size = 64
        model_args.block_size = 512
        return model_args


if __name__ == "__main__":
    test = LM_trainer()
    test.train()