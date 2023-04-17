# Based on notebook from HuggingFace:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=DDtsaJeVIrJT

import transformers
from datasets import load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


from utils import load_into_datasetdict, tokenize_and_align_labels, compute_metrics

class something():
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
        
        # Initialize the tokenizer and model 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)

        # Tokenize and align the labels on a sub-word level for all datasets
        self.tokenized_datasets = self.datasets.map(lambda examples: tokenize_and_align_labels(examples=examples, tokenizer=self.tokenizer, label_all_tokens=self.label_all_tokens), batched=True)

    def set_trainer(self, use_old = False):
        if use_old:
            self.old_model()
        else: 
            self.new_model()

        # Arguments for the trainer object
        args = TrainingArguments(
            f"{self.model_name}-finetuned-{self.task}",
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=3,
            weight_decay=0.01
        )

        # Pad the labels to the maximum length of the sequences in the examples given
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Load the metrics function
        metric = load_metric("seqeval")

        example = self.tokenized_datasets["train"][17]

        labels = [self.label_list[i] for i in example[f"tags"]]
        print(metric.compute(predictions=[labels], references=[labels]))

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
    model_name = "bert-base-multilingual-cased"
    batch_size = 32

    # Flag to indicate whether to label all tokens or just the first token of each word
    label_all_tokens = True

    # File paths to splits of the chosen dataset
    file_paths = {
        "train": "data/baseline/en_ewt_nn_email_dev.conll",
        "validation": "data/baseline/en_ewt_nn_answers_dev.conll",
        "test": "data/baseline/en_ewt_nn_answers_test.conll",
    }

    s = something(task, model_name, batch_size, label_all_tokens, file_paths)
    s.train_and_save()

    print(s.evaluate())

    # load trianed model to trainer
    s.set_trainer(use_old = True)
    print(s.evaluate())