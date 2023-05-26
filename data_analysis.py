import pandas as pd
import numpy as np
from collections import Counter
from utils import read_conll

paths  = {'en': ['data/datasets/ewt/en_ewt_nn_train.conll', 'data/datasets/ewt/en_ewt_nn_newsgroup_test.conll', 'data/datasets/ewt/en_ewt_nn_newsgroup_dev.conll'], 
                  'ger': "data/datasets/NoSta-D/NER-de-test.tsv", 
                  'dk': "data/datasets/DaNplus/da_news_test.tsv"}


def get_data(paths):
    if type(paths) == list:
        labels = []
        for path in paths:
            _, l = read_conll(path)
            labels += l
    else:
        _, labels = read_conll(paths)
    labels = [item.replace("B-", "").replace("I-", "") for item in labels for item in item]
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
    # print(pd.DataFrame.from_dict(all_datasets(english_paths)).to_latex())
    print(all_datasets(paths))
    ## get a dictionary of type {train:{count:{}, ratio:{}}}