import datasets
import pandas as pd
from datasets import Dataset, load_dataset

from utils import read_conll

train_tokens, train_tags = read_conll("data/baseline/en_ewt_nn_train_answers_only.conll")
dev_tokens, dev_tags = read_conll("data/baseline/en_ewt_nn_answers_dev.conll")
test_tokens, test_tags = read_conll("data/baseline/en_ewt_nn_answers_test.conll")

def convert_to_dataset(tokens, tags):
    df = pd.DataFrame({"tokens": tokens, "tags": tags})
    df['id'] = df.reset_index().index
    df = df[['id', 'tokens', 'tags']]
    dataset = Dataset.from_pandas(df)

    return dataset


train_dataset = convert_to_dataset(train_tokens, train_tags)
dev_dataset = convert_to_dataset(dev_tokens, dev_tags)
test_dataset = convert_to_dataset(test_tokens, test_tags)

# convert to datasetdict
datasets = datasets.DatasetDict({
    "train": train_dataset,
    "validation": dev_dataset,
    "test": test_dataset
})


print(datasets)