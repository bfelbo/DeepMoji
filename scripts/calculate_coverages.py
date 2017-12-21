from __future__ import print_function
import pickle
import json
import csv
import sys

# Allow us to import the deepmoji directory
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from deepmoji.sentence_tokenizer import SentenceTokenizer, coverage

OUTPUT_PATH = 'coverage.csv'
DATASET_PATHS = [
    '../data/Olympic/raw.pickle',
    '../data/PsychExp/raw.pickle',
    '../data/SCv1/raw.pickle',
    '../data/SCv2-GEN/raw.pickle',
    '../data/SE0714/raw.pickle',
    # '../data/SE1604/raw.pickle', # Excluded due to Twitter's ToS
    '../data/SS-Twitter/raw.pickle',
    '../data/SS-Youtube/raw.pickle',
]

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

results = []
for p in DATASET_PATHS:
    coverage_result = [p]
    print('Calculating coverage for {}'.format(p))
    with open(p) as f:
        s = pickle.load(f)

    # Decode data
    try:
        s['texts'] = [unicode(x) for x in s['texts']]
    except UnicodeDecodeError:
        s['texts'] = [x.decode('utf-8') for x in s['texts']]

    # Own
    st = SentenceTokenizer({}, 30)
    tests, dicts, _ = st.split_train_val_test(s['texts'], s['info'],
                                              [s['train_ind'],
                                               s['val_ind'],
                                               s['test_ind']],
                                              extend_with=10000)
    coverage_result.append(coverage(tests[2]))

    # Last
    st = SentenceTokenizer(vocab, 30)
    tests, dicts, _ = st.split_train_val_test(s['texts'], s['info'],
                                              [s['train_ind'],
                                               s['val_ind'],
                                               s['test_ind']],
                                              extend_with=0)
    coverage_result.append(coverage(tests[2]))

    # Full
    st = SentenceTokenizer(vocab, 30)
    tests, dicts, _ = st.split_train_val_test(s['texts'], s['info'],
                                              [s['train_ind'],
                                               s['val_ind'],
                                               s['test_ind']],
                                              extend_with=10000)
    coverage_result.append(coverage(tests[2]))

    results.append(coverage_result)

with open(OUTPUT_PATH, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n')
    writer.writerow(['Dataset', 'Own', 'Last', 'Full'])
    for i, row in enumerate(results):
        try:
            writer.writerow(row)
        except Exception:
            print("Exception at row {}!".format(i))

print('Saved to {}'.format(OUTPUT_PATH))
