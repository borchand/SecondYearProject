# Based on notebook from HuggingFace:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=DDtsaJeVIrJT

import os
import re
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import transformers
from evaluate import load as load_metric
from torch.optim import AdamW
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils import (
    compute_metrics,
    convert_to_dataset,
    get_optimizer_params,
    load_into_datasetdict,
    read_conll,
    tokenize_and_align_labels,
)


class MyTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs)



class TokenClassificationTrainer():
    def __init__(self, task, model_name, save_name, batch_size, label_all_tokens, file_paths):
        self.task = task
        self.model_name = model_name
        self.save_name = save_name
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

    def set_trainer(self, use_old = False, learning_rate=2e-6, num_epochs = 80, weight_decay = 0.01, checkpoint_path = "", plotting=False, discriminate_lr=False, seed=123, rate = 0.7):
        if use_old:
            self.old_model()
        else: 
            self.new_model()

        # Create folder for trainer checkpoints if not already existing
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        # Arguments for the trainer object
        args = TrainingArguments(
            f"checkpoints/{self.model_name}-finetuned-{self.task}-{self.save_name}",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            save_total_limit=1,
            learning_rate=learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            use_mps_device=False,
            load_best_model_at_end = True,
            metric_for_best_model = 'span_f1',
            greater_is_better = True,
            seed=seed
        )

        # Pad the labels to the maximum length of the sequences in the examples given
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Load the metrics function
        metric = load_metric("seqeval")

        example = self.tokenized_datasets["train"][17]

        labels = [self.label_list[i] for i in example[f"tags"]]

        parameters = get_optimizer_params(self.model, learning_rate=learning_rate, rate=rate)
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }

        optimizer = AdamW(parameters if discriminate_lr else self.model.parameters(), lr=learning_rate, **kwargs)

        # plotting
        if plotting and discriminate_lr:
            learning_rates1, learning_rates2, learning_rates3, learning_rates4, learning_rates5, learning_rates6 = [[] for i in range(6)]
            for i in range(num_epochs):
                optimizer.step()
                learning_rates1.append(optimizer.param_groups[0]["lr"])
                learning_rates2.append(optimizer.param_groups[1]["lr"])
                learning_rates3.append(optimizer.param_groups[2]["lr"])
                learning_rates4.append(optimizer.param_groups[3]["lr"])
                learning_rates5.append(optimizer.param_groups[4]["lr"])
                learning_rates6.append(optimizer.param_groups[10]["lr"])

            plt.plot(learning_rates1, label="Embeddings")
            plt.plot(learning_rates2, label="Transformer layer 0-3")
            plt.plot(learning_rates3, label="Transformer layer 4-7")
            plt.plot(learning_rates4, label="Transformer layer 8-11")
            plt.plot(learning_rates5, label="Transformer layer 12-15")
            plt.plot(learning_rates6, label="Classifier")
            plt.yscale("log")        
            plt.legend()
            plt.show()



        self.trainer = MyTrainer(
            self.model,
            args,
            optimizers=(optimizer, None),
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=lambda p: compute_metrics(p=p, label_list=self.label_list, metric=metric),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        return self.trainer
    

    def new_model(self):
        # Load in the model from the pretrained checkpoint
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(self.label_list))
    
    def old_model(self):
        # Save the model
        self.model = AutoModelForTokenClassification.from_pretrained(f"models/{self.model_name}-finetuned-{self.task}-{self.save_name}")

    def train(self, **kwargs):
        self.trainer = self.set_trainer(use_old=False, **kwargs)
        self.trainer.train()

        return self.trainer
    
    def train_and_save(self, **kwargs):
        self.trainer = self.set_trainer(use_old=False, **kwargs)
        self.trainer.train()
        self.trainer.save_model(f"models/{self.model_name}-finetuned-{self.task}-{self.save_name}")

        return self.trainer
    
    def evaluate(self):
        return self.trainer.evaluate()
    
    def evaluate_multiple(self, paths):
        if type(paths) == str:
            paths = [paths]

        evaluations = []
        for path in paths:
            print(path)
            tokens, tags = read_conll(path)
            dataset = convert_to_dataset(tokens, tags)
            tokenized_dataset = dataset.map(lambda examples: tokenize_and_align_labels(examples=examples, tokenizer=self.tokenizer, label_all_tokens=self.label_all_tokens, fast=self.fast), batched=True)
            evaluation = self.trainer.evaluate(eval_dataset=tokenized_dataset)
            print(evaluation)
            evaluations.append(evaluation)
        return evaluations

    def del_trainer(self):
        del self.trainer
        del self.tokenizer
        del self.model
        del self.datasets

    def predict(self, path):
        tokens, tags = read_conll(path)
        dataset = convert_to_dataset(tokens, tags)
        tokenized_dataset = dataset.map(lambda examples: tokenize_and_align_labels(examples=examples, tokenizer=self.tokenizer, label_all_tokens=self.label_all_tokens, fast=self.fast), batched=True)
        predictions, labels, _ = self.trainer.predict(tokenized_dataset)
        return predictions, labels




if __name__ == "__main__":
    # Set the task and name of the pretrained model and the batch size for finetuning
    task = "ner"
    model_name = "xlm-mlm-17-1280"  # "bert-base-multilingual-cased" or "xlm-mlm-17-1280"
    seed = 57808
    save_name = "discriminate-lr.seed-" + str(seed)
    batch_size = 32

    # Flag to indicate whether to label all tokens or just the first token of each word
    label_all_tokens = False

    # File paths to splits of the chosen dataset
    file_paths = {
        "train": "data/datasets/ewt/en_ewt_nn_train.conll",
        "validation": "data/datasets/ewt/en_ewt_nn_newsgroup_dev.conll",
        "test": "data/datasets/ewt/en_ewt_nn_test_newsgroup_and_weblogs.conll",
    }

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the datasets into a DatasetDict
    datasets = load_into_datasetdict(file_paths)
    tokenized_datasets = datasets.map(lambda examples: tokenize_and_align_labels(examples=examples, tokenizer=tokenizer, label_all_tokens=label_all_tokens, fast=False), batched=True)

    tokenClassificationTrainer = TokenClassificationTrainer(task, model_name, save_name, batch_size, label_all_tokens, file_paths)

    # load trianed model to trainer
    tokenClassificationTrainer.set_trainer(use_old = True)

    # Evaluate on the test set

    # Get the label names from the datasets
    label_list = datasets["train"].features["tags"].feature.names

    # Save to preds.conll
    predictions, labels = tokenClassificationTrainer.predict(file_paths["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    punct_regex = re.compile(r"^\W$")

    with open("preds.conll", "w") as f:
        for sent, preds, labels in zip(tokenized_datasets["test"]["tokens"], true_predictions, true_labels):
            sent = [word for word in sent if not punct_regex.match(word)]
            if len(sent) != len(preds) or len(sent) != len(labels):
                continue
            for word, pred, label in zip(sent, preds, labels):
                msg = f"{word}\t{pred}\t{label}\n"
                f.write(msg)
            f.write("\n")
