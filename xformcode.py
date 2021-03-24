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

def do_nlp_fnx(row):
    global nlp
    try:
        if 'sequence' in row:
            s = nlp(row['sequence'])
        elif 'text' in row:
            s = nlp(row['text'])
        else:
            return '', ''
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
        return [org_name, per_name]

# This transform is called for each file in the chosen data
def infin_transform_one_object(filename, output_dir, parentdir, **kwargs):
    print('infin_transform_one_object: Entered. filename=' + filename + ', output_dir=' + output_dir)

    if (filename.endswith('.json')):
        df = pd.read_json(filename, orient='records', typ='frame')
        df[['organization', 'person']] = df.apply(do_nlp_fnx, axis=1, result_type='expand')
    else:
        arr = []
        inf = open(filename, 'r', errors='ignore')
        for line in inf.readlines():
            do_nlp(line, arr)
        df = pd.DataFrame(arr, columns=['text', 'organization', 'person'])

    print('infin_transform_one_object: shape=' + str(df.shape), flush=True)
    cols = df.columns.values.tolist()
    print('infin_transform_one_object: columns = ' + str(cols), flush=True)
    for index, rw in df.iterrows():
        print('infin_transform_one_object: row = ' + str(rw), flush=True)
    print('infin_transform_one_object: finished creating dataframe', flush=True)
    df.to_json(filename + '.json', orient='records')
    print('infin_transform_one_object: finished writing df to json. file=' + filename + '.json', flush=True)
    log_artifact(filename + '.json', parentdir)
    print('infin_transform_one_object: finished logging artifact', flush=True)
