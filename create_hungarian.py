import pandas as pd
import os
import numpy as np

path = './NYTK-NerKor/data/genres/news/no-morph'
directory = os.listdir('./NYTK-NerKor/data/genres/news/no-morph')


with open('hungarian_dataset.tsv', 'w') as f:
    for filename in directory:
        print(filename)
        filepath = os.path.join(path, filename)
        data = pd.read_csv(filepath, sep='\t', skiprows=1, names=['word', 'pos', 'lemma', 'morph','pass', 'ner'], skip_blank_lines=False, quoting=3)
        data = data.replace(np.nan, '', regex=True)
        print(data.head(30))
        ## itterate over rows
        x = 0
        for row in range(data.shape[0]):
            if x%10 == 0:
                print(row)
            if data['word'][row] == '':
                f.write('\n')
            elif '  ' in data['word'][row]:
                continue
            else:
                f.write(data['word'][row]+ '    ' + data['ner'][row] + '\n')
            x += 1
        