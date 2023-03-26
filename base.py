from datasets import load_dataset, Dataset
import pandas as pd
from utils import read_conll

tokens, tags = read_conll("./data/baseline/en_ewt_nn_answers_dev.conll")

df = pd.DataFrame({"tokens": tokens, "tag": tags})
df['id'] = df.reset_index().index

dataset = Dataset.from_pandas(df)
dataset = Dataset.train_test_split(dataset, train_size=0.8)

print(dataset)
print(dataset["train"][0])

# dataset = load_dataset("lhoestq/demo1")

# print(dataset["train"][0])