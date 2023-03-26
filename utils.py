# Utility functions for the NER model.
import datasets
import pandas as pd
from datasets import Dataset


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
    }
)


def convert_to_dataset(tokens, tags, features=CONLL_FEATURES):
    df = pd.DataFrame({"tokens": tokens, "tags": tags})
    df['id'] = df.reset_index().index
    df = df[['id', 'tokens', 'tags']]
    dataset = Dataset.from_pandas(df, features=features)

    return dataset


def load_into_datasetdict(path_dict, features=CONLL_FEATURES):
    dataset_splits = dict()

    for key, path in path_dict.items():
        tokens, tags = read_conll(path)
        dataset = convert_to_dataset(tokens, tags, features=features)

        dataset_splits[key] = dataset

    return datasets.DatasetDict(dataset_splits, features=features)