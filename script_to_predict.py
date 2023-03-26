import pandas as pd
import codecs
import pickle  as pickle
from transformers import AutoModelForTokenClassification, AutoTokenizer

data = []
current_words = []
current_tags = []
labels = []



for line in codecs.open('data/baseline/en_ewt_nn_answers_test.conll', encoding='utf-8'):
    line = line.strip()

    if line:
        if line[0] == '#':
            continue # skip comments
        tok = line.split('\t')
        word = tok[0]
        tag = tok[1]
        tag2 = tok[2] ## Not used for now, but here for future use if needed

        current_words.append(word)
        current_tags.append(tag)
    else:
        if current_words:  # skip empty lines
            data.append((current_words, current_tags))
        current_words = []
        current_tags = []


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

## Find all the unique labels in the data


## NAME CSV WHAT YOU WANT AND GIVE IT A PATH MABY??
outF = open("baseline.txt", "w")


model = AutoModelForTokenClassification.from_pretrained("models/bert-base-multilingual-cased-finetuned-ner")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

## get all the unique labels from the data in a list


for i in data:
    sent = i[0]

    pa = tokenizer.encode(' '.join(sent), return_tensors="pt", add_special_tokens=False)
    predictions = model(pa)

    for index in range(1, len(sent) + 1):
        pred = predictions.logits[0][index].argmax().item()
        
        
        print(pred)

        ## HERE word[index] is the original word from the sentanve, and pred[index] is the predicted label for that
        out = sent[index-1] + '\t' + names[pred]

        # wrtie line to output file
        outF.write(out)
        outF.write("\n")
    break
    out = '\t'
    outF.write(out)
    outF.write("\n")
