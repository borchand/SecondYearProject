import pandas as pd
import numpy as np
from collections import Counter
from utils import read_conll

english_paths  = {'train': 'data/datasets/baseline/en_ewt_nn_train.conll', 
                  'validation': 'data/datasets/baseline/en_ewt_nn_newsgroup_test.conll', 
                  'test': 'data/datasets/baseline/en_ewt_nn_newsgroup_dev.conll'}

german_paths = {"train": "data/datasets/NoSta-D/NER-de-train.tsv",
                "validation": "data/datasets/NoSta-D/NER-de-test.tsv",
                "test": "data/datasets/NoSta-D/NER-de-dev.tsv",}

danish_paths = { "train": "data/datasets/DaNplus/da_news_train.tsv",
                "validation": "data/datasets/DaNplus/da_news_test.tsv",
                "test": "data/datasets/DaNplus/da_news_dev.tsv",}


def get_data(path):
    _, labels = read_conll(path)
    labels = [item for item in labels for item in item]
    count = Counter(labels)
    sum_ = sum(count.values())
    ratio  = {}
    for i in count.keys():
        ratio[i] = count[i]/sum_
    return count, ratio


def all_datasets(paths, specific = False):
    if specific != False:
        data = get_data(paths[specific])
        return data
    
    data = {}
    for i in paths.keys():
        temp_ = {}
        print(paths[i])
        count, ratio = get_data(paths[i])
        temp_['count'] = count
        temp_['ratio'] = ratio
        data[i] = temp_

    return data

        

        
if __name__ == '__main__':
    print(all_datasets(english_paths))
    ## get a dictionary of type {train:{count:{}, ratio:{}}}