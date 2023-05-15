import pandas as pd
import os
import numpy as np

# path = './NYTK-NerKor/data/genres/news/no-morph'
# directory = os.listdir('./NYTK-NerKor/data/genres/news/no-morph')


# with open('/data/datasets/hungarian_dataset.tsv', 'w') as f:
#     for filename in directory:
#         print(filename)
#         filepath = os.path.join(path, filename)
#         data = pd.read_csv(filepath, sep='\t', skiprows=1, names=['word', 'pos', 'lemma', 'morph','pass', 'ner'], skip_blank_lines=False, quoting=3)
#         data = data.replace(np.nan, '', regex=True)
#         print(data.head(30))
#         ## itterate over rows
#         x = 0
#         for row in range(data.shape[0]):
#             if x%10 == 0:
#                 print(row)
#             if data['word'][row] == '':
#                 f.write('\n')
#             elif '  ' in data['word'][row]:
#                 continue
#             else:
#                 f.write(data['word'][row]+ '    ' + data['ner'][row] + '\n')
#             x += 1


hungarian = pd.read_csv('data/datasets/hungarian/hungarian_dataset.tsv', sep='    ', skip_blank_lines=False, quoting=3, header=0)
hungarian  = hungarian.replace(np.nan, '', regex=True)
full_thing = []
sent = []
ner = []
for i in range(hungarian.shape[0]):
    if hungarian['word'][i] == '':
        full_thing.append([sent, ner])
        sent = []
        ner = []
    else:
        sent.append(hungarian['word'][i])
        ner.append(hungarian['ner'][i])
        
## split full_thing into train, test, validation
np.random.seed(42)
np.random.shuffle(full_thing)
train = full_thing[:int(len(full_thing)*0.8)]
test = full_thing[int(len(full_thing)*0.8):int(len(full_thing)*0.9)]
validation = full_thing[int(len(full_thing)*0.9):]

## write to file
with open('data/datasets/hungarian/hungarian_train.tsv', 'w') as f:
    for sent in train:
        for i in range(len(sent[0])):
            f.write(sent[0][i] + '\t' + sent[1][i] + '\n')
        f.write('\n')
        
with open('data/datasets/hungarian/hungarian_test.tsv', 'w') as f:
    for sent in test:
        for i in range(len(sent[0])):
            f.write(sent[0][i] + '\t' + sent[1][i] + '\n')
        f.write('\n')
        
with open('data/datasets/hungarian/hungarian_validation.tsv', 'w') as f:
    for sent in validation:
        for i in range(len(sent[0])):
            f.write(sent[0][i] + '\t' + sent[1][i] + '\n')
        f.write('\n')
