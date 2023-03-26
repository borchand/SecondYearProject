import pandas as pd

from utils import load_into_datasetdict

paths = {
    "train": "data/baseline/en_ewt_nn_train_answers_only.conll",
    "validation": "data/baseline/en_ewt_nn_answers_dev.conll",
    "test": "data/baseline/en_ewt_nn_answers_test.conll",
}

datasets = load_into_datasetdict(paths)


print(datasets["train"])