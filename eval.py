# based on:
# https://github.itu.dk/robv/intro-nlp2023/blob/main/assignments/project/span_f1.py

import sys

def readBIO(path):
    ents = []
    words = []
    curEnts = []
    curWords = []
    for line in open(path):
        line = line.strip()
        if line == '':
            ents.append(curEnts)
            words.append(curWords)
            curEnts = []
            curWords = []
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        else:
            curEnts.append(line.split('\t')[1])
            curWords.append(line.split('\t')[0])
    return ents, words

def toSpans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg+1, len(tags)):
                if tags[beg][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans

def getInstanceScores(predPath, goldPath):
    goldEnts, goldWords = readBIO(goldPath)
    predEnts, predWords = readBIO(predPath)
    entScores = []
    tp = 0
    fp = 0
    fn = 0

    for goldEnt, goldWord, predEnt, predWord in zip(goldEnts, goldWords, predEnts, predWords):
        goldSpans = toSpans(goldEnt)
        predSpans = toSpans(predEnt)

        overlap = len(goldSpans.intersection(predSpans))
        tp += overlap
        fp += len(predSpans) - overlap
        fn += len(goldSpans) - overlap
        
    prec = 0.0 if tp+fp == 0 else tp/(tp+fp)
    rec = 0.0 if tp+fn == 0 else tp/(tp+fn)
    f1 = 0.0 if prec+rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    return f1
    
    

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('please provide path to gold file and output of your system (in same format)')
        print('for example: \npython3 eval.py bert_out-dev.conll opener_en-dev.conll')
    else:
        score = getInstanceScores(sys.argv[1], sys.argv[2])
        print(score)