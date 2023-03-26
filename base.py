# Based on notebook from HuggingFace:
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=DDtsaJeVIrJT

import numpy as np
import transformers
from datasets import load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from utils import load_into_datasetdict

# Set the task and name of the pretrained model and the batch size for finetuning
task = "ner"
model_name = "bert-base-multilingual-cased"
batch_size = 32

# Flag to indicate whether to label all tokens or just the first token of each word
label_all_tokens = True

# File paths to splits of the chosen dataset
file_paths = {
    "train": "data/baseline/en_ewt_nn_train.conll",
    "validation": "data/baseline/en_ewt_nn_answers_dev.conll",
    "test": "data/baseline/en_ewt_nn_answers_test.conll",
}

# Load the datasets into a DatasetDict
datasets = load_into_datasetdict(file_paths)

# Get the label names from the datasets
label_list = datasets["train"].features["tags"].feature.names

# Initialize the tokenizer and model 
tokenizer = AutoTokenizer.from_pretrained(model_name)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

# Function to tokenize and align the labels on a sub-word level
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize and align the labels on a sub-word level for all datasets
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# Load in the model from the pretrained checkpoint
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

# Arguments for the trainer object
args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01
)

# Pad the labels to the maximum length of the sequences in the examples given
data_collator = DataCollatorForTokenClassification(tokenizer)

# Load the metrics function
metric = load_metric("seqeval")

example = tokenized_datasets["train"][17]

labels = [label_list[i] for i in example[f"tags"]]
print(metric.compute(predictions=[labels], references=[labels]))


def compute_metrics(p):
    predictions, labels = p
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

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print(trainer.train())

print(trainer.evaluate())


# Save the model
trainer.save_model(f"models/{model_name}-finetuned-{task}")
