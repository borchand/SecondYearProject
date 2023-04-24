# Utility functions for the NER model.
import datasets
import pandas as pd
from datasets import Dataset
import numpy as np

def read_conll(path, nested=False):
    """Reads a CoNLL file and returns a list of sentences and their corresponding labels.

    Args:
        path (str): Path to the CoNLL file.

    Returns:
        sents: List of sentences.
        labels: List of labels for each sentence.
    """
    sents = []
    labels = []
    if nested:
        nested_labels = []

    with open(path, 'r') as f:
        raw_sents = f.read().split("\n\n")
    
    for raw_sent in raw_sents:
        text = []
        label = []
        if nested:
            nested_label = []
        for line in raw_sent.split("\n"):
            if line != "":
                line = line.split("\t")
                text.append(line[0])
                label.append(line[1])
                if nested:
                    nested_label.append(line[2])
        sents.append(text)
        labels.append(label)
        if nested:
            nested_labels.append(nested_label)
    
    if nested:
        return sents, labels, nested_labels

    return sents, labels


CONLL_FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "tags": datasets.Sequence(
            datasets.features.ClassLabel(
                names=[
                    "O",
                    "B-PER",
                    "I-PER",
                    "B-ORG",
                    "I-ORG",
                    "B-LOC",
                    "I-LOC",
                    "B-MISC",
                    "I-MISC",
                ]
            )
        ),
        "text": datasets.Value("string"),
    }
)


def convert_to_dataset(tokens, tags, features=CONLL_FEATURES):
    df = pd.DataFrame({"tokens": tokens, "tags": tags, "text": [' '.join(token) for token in tokens]})
    df['id'] = df.reset_index().index
    df = df[['id', 'tokens', 'tags', 'text']]
    dataset = Dataset.from_pandas(df, features=features)

    return dataset


def load_into_datasetdict(path_dict, features=CONLL_FEATURES):
    dataset_splits = dict()

    for key, path in path_dict.items():
        tokens, tags = read_conll(path)
        dataset = convert_to_dataset(tokens, tags, features=features)

        dataset_splits[key] = dataset

    return datasets.DatasetDict(dataset_splits)

# Function to tokenize and align the labels on a sub-word level
def tokenize_and_align_labels(examples, tokenizer, label_all_tokens, fast):

    tokenized_inputs = tokenizer(examples["text"])
    # examples["text"]
    # print(tokenized_inputs)
    for text, ids in zip(examples["text"], tokenized_inputs["input_ids"]):
        print(text)
        print(tokenizer.convert_ids_to_tokens(ids))
        print("\n")
    labels = []
    for i, label in enumerate(examples[f"tags"]):
        if fast:
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

def compute_metrics(p, label_list, metric):
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