# Based on notebook from HuggingFace:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=DDtsaJeVIrJT

import transformers
from evaluate import load as load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from utils import compute_metrics, load_into_datasetdict, tokenize_and_align_labels


class TokenClassificationTrainer():
    def __init__(self, task, model_name, batch_size, label_all_tokens, file_paths):
        self.task = task
        self.model_name = model_name
        self.batch_size = batch_size
        self.label_all_tokens = label_all_tokens
        self.file_paths = file_paths

        # Load the datasets into a DatasetDict
        self.datasets = load_into_datasetdict(self.file_paths)
        # Get the label names from the datasets
        self.label_list = self.datasets["train"].features["tags"].feature.names

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize the tokenizer and model 
        if "xlm" in self.model_name:
            self.fast = False
            assert isinstance(self.tokenizer, transformers.XLMTokenizer)
        else:
            self.fast = True
            assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)

        # Tokenize and align the labels on a sub-word level for all datasets
        self.tokenized_datasets = self.datasets.map(lambda examples: tokenize_and_align_labels(examples=examples, tokenizer=self.tokenizer, label_all_tokens=self.label_all_tokens, fast=self.fast), batched=True)

    def set_trainer(self, use_old = False, learning_rate=2e-5, num_train_epochs = 3, weight_decay = 0.01):
        if use_old:
            self.old_model()
        else: 
            self.new_model()

        # Arguments for the trainer object
        args = TrainingArguments(
            f"{self.model_name}-finetuned-{self.task}",
            evaluation_strategy = "epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            use_mps_device=False
        )

        # Pad the labels to the maximum length of the sequences in the examples given
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Load the metrics function
        metric = load_metric("seqeval")

        example = self.tokenized_datasets["train"][17]

        labels = [self.label_list[i] for i in example[f"tags"]]

        self.trainer = Trainer(
            self.model,
            args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=lambda p: compute_metrics(p=p, label_list=self.label_list, metric=metric)
        )

        return self.trainer
    

    def new_model(self):
        # Load in the model from the pretrained checkpoint
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(self.label_list))
    
    def old_model(self):
        # Save the model
        self.model = AutoModelForTokenClassification.from_pretrained(f"models/{self.model_name}-finetuned-{self.task}")

    def train_and_save(self):
        self.trainer = self.set_trainer(use_old=False)
        self.trainer.train()
        self.trainer.save_model(f"models/{model_name}-finetuned-{task}")

        return self.trainer
    
    def evaluate(self):
        return self.trainer.evaluate()


if __name__ == "__main__":
    # Set the task and name of the pretrained model and the batch size for finetuning
    task = "ner"
    model_name = "xlm-mlm-17-1280"  # "bert-base-multilingual-cased" or "xlm-mlm-17-1280"
    batch_size = 16

    # Flag to indicate whether to label all tokens or just the first token of each word
    label_all_tokens = True

    # File paths to splits of the chosen dataset
    file_paths = {
        "train": "data/datasets/baseline/en_ewt_nn_train.conll",
        "validation": "data/datasets/baseline/en_ewt_nn_newsgroup_dev.conll",
        "test": "data/datasets/baseline/en_ewt_nn_newsgroup_test.conll",
    }

    tokenClassificationTrainer = TokenClassificationTrainer(task, model_name, batch_size, label_all_tokens, file_paths)
    tokenClassificationTrainer.train_and_save()

    print(tokenClassificationTrainer.evaluate())

    # load trianed model to trainer
    tokenClassificationTrainer.set_trainer(use_old = True)
    print(tokenClassificationTrainer.evaluate())