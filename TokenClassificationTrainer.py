# Based on notebook from HuggingFace:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=DDtsaJeVIrJT

import matplotlib.pyplot as plt
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
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from utils import (
    compute_metrics,
    convert_to_dataset,
    get_optimizer_params,
    load_into_datasetdict,
    read_conll,
    tokenize_and_align_labels,
)


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

    def set_trainer(self, use_old = False, learning_rate=2e-5, num_train_epochs = 10, weight_decay = 0.01, scheduler = True, checkpoint_path = "", plotting=False, discriminate_lr=False):
        if use_old:
            self.old_model()
        else: 
            self.new_model()

        # Arguments for the trainer object
        args = TrainingArguments(
            f"{checkpoint_path}{self.model_name}-finetuned-{self.task}-{self.save_name}",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            save_total_limit=1,
            learning_rate=learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            use_mps_device=False,
            load_best_model_at_end = True
        )

        # Pad the labels to the maximum length of the sequences in the examples given
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Load the metrics function
        metric = load_metric("seqeval")

        example = self.tokenized_datasets["train"][17]

        labels = [self.label_list[i] for i in example[f"tags"]]

        parameters = get_optimizer_params(self.model, learning_rate=learning_rate)
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }

        optimizer = AdamW(parameters if discriminate_lr else self.model.parameters(), lr=learning_rate, **kwargs)
        if scheduler:
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_train_epochs // 3, num_training_steps=num_train_epochs)
        else:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_train_epochs // 3)


        # plotting
        if plotting and discriminate_lr:
            learning_rates1, learning_rates2, learning_rates3, learning_rates4, learning_rates5, learning_rates6 = [[] for i in range(6)]
            for i in range(num_train_epochs):
                optimizer.step()
                scheduler.step()
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



        self.trainer = Trainer(
            self.model,
            args,
            # optimizers=(None, None),
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=lambda p: compute_metrics(p=p, label_list=self.label_list, metric=metric),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        return self.trainer
    

    def new_model(self):
        # Load in the model from the pretrained checkpoint
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(self.label_list))
    
    def old_model(self):
        # Save the model
        self.model = AutoModelForTokenClassification.from_pretrained(f"models/{self.model_name}-finetuned-{self.task}")

    def train_and_save(self, **kwargs):
        self.trainer = self.set_trainer(use_old=False, **kwargs)
        self.trainer.train()
        self.trainer.save_model(f"models/{self.model_name}-finetuned-{self.task}")

        return self.trainer
    
    def evaluate(self):
        return self.trainer.evaluate()
    
    def evaluate_multiple(self, paths):
        if type(paths) == str:
            paths = [paths]

        evaluations = []
        for path in paths:
            tokens, tags = read_conll(path)
            dataset = convert_to_dataset(tokens, tags)
            tokenized_dataset = dataset.map(lambda examples: tokenize_and_align_labels(examples=examples, tokenizer=self.tokenizer, label_all_tokens=self.label_all_tokens, fast=self.fast), batched=True)
            evaluation = self.trainer.evaluate(eval_dataset=tokenized_dataset)
            print(evaluation)
            evaluations.append(evaluation)
        return evaluations




if __name__ == "__main__":
    # Set the task and name of the pretrained model and the batch size for finetuning
    task = "ner"
    model_name = "xlm-mlm-17-1280"  # "bert-base-multilingual-cased" or "xlm-mlm-17-1280"
    batch_size = 32

    # Flag to indicate whether to label all tokens or just the first token of each word
    label_all_tokens = True

    # File paths to splits of the chosen dataset
    file_paths = {
        "train": "data/datasets/baseline/en_ewt_nn_train.conll",
        "validation": "data/datasets/baseline/en_ewt_nn_newsgroup_dev.conll",
        "test": "data/datasets/baseline/en_ewt_nn_newsgroup_test.conll",
    }

    tokenClassificationTrainer = TokenClassificationTrainer(task, model_name, batch_size, label_all_tokens, file_paths)

    # load trianed model to trainer
    tokenClassificationTrainer.set_trainer(use_old = False)
    # print(tokenClassificationTrainer.evaluate())

