import pandas as pd
import codecs





data = []
current_words = []
current_tags = []

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


## NAME CSV WHAT YOU WANT AND GIVE IT A PATH MABY??
outF = open("baseline.txt", "w")

for i in data:
    sent = i[0]
    label = i[1]
    
    ## Do some fancy predictioning here
    
    
    
    
    ## pred = predictions made for each word in sent
    
    for index in range(len(sent)):
        
        ## HERE word[index] is the original word from the sentanve, and pred[index] is the predicted label for that
        out = sent[index] + '\t' + label[index] 

        # wrtie line to output file
        outF.write(out)
        outF.write("\n")
        
    out = '\t'
    outF.write(out)
    outF.write("\n")

        