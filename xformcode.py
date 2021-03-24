from transformers import pipeline
import pandas as pd
import os
from mlflow import log_artifact
import gc

nlp = pipeline('ner')

def do_nlp(line, arr):
    try:
        global nlp
        s = nlp(line)
    except Exception as err:
        print(f'error occurred while nlp: {err}')
    else:
        org_name = ''
        per_name = ''
        for one_word in s:
            if (one_word['entity'] == 'I-ORG'):
                if (one_word['word'].startswith('##')):
                    org_name = org_name + one_word['word'][2:]
                else:
                    org_name = org_name + ' ' + one_word['word']
            if (one_word['entity'] == 'I-PER'):
                if (one_word['word'].startswith('##')):
                    per_name = per_name + one_word['word'][2:]
                else:
                    per_name = per_name + ' ' + one_word['word']
        arr.append([line, org_name, per_name])

# This transform is called for each file in the chosen data
def infin_transform_one_object(filename, output_dir, parentdir, **kwargs):
    print('infin_transform_one_object: Entered. filename=' + filename + ', output_dir=' + output_dir)

    arr = []
    if (filename.endswith('.json')):
        indf = pd.read_json(filename, 'records', 'frame')
        if 'sequence' in indf:
            for index, one_row in indf.iterrows():
                do_nlp(one_row['sequence'], arr)
        elif 'text' in indf:
            for index, one_row in indf.iterrows():
                do_nlp(one_row['text'], arr)
        else:
            print('infin_transform_one_object: Do not know which column has text', flush=True)
            return
    else:
        inf = open(filename, 'r', errors='ignore')
        for line in inf.readlines():
            do_nlp(line, arr)

    df = pd.DataFrame(arr, columns=['text', 'organization', 'person'])
    print('infin_transform_one_object: finished creating dataframe', flush=True)
    df.to_json(filename + '.json', orient='records')
    print('infin_transform_one_object: finished writing df to json. file=' + filename + '.json', flush=True)
    log_artifact(filename + '.json', parentdir)
    print('infin_transform_one_object: finished logging artifact', flush=True)
    del arr
    del[[df]]
    gc.collect()

