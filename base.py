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

# Load the datasets into a DatasetDict
datasets = load_into_datasetdict(file_paths)

# Get the label names from the datasets
label_list = datasets["train"].features["tags"].feature.names

# Initialize the tokenizer and model 
tokenizer = AutoTokenizer.from_pretrained(model_name)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)



# Tokenize and align the labels on a sub-word level for all datasets
tokenized_datasets = datasets.map(lambda examples: tokenize_and_align_labels(examples=examples, tokenizer=tokenizer, label_all_tokens=label_all_tokens), batched=True)

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


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=lambda p: compute_metrics(p=p, label_list=label_list, metric=metric)
)

print(trainer.train())

print(trainer.evaluate())


# Save the model
trainer.save_model(f"models/{model_name}-finetuned-{task}")