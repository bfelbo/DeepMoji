from __future__ import print_function

import json
import math
import pickle
import sys
import numpy as np
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from deepmoji.word_generator import WordGenerator
from deepmoji.create_vocab import VocabBuilder
from deepmoji.sentence_tokenizer import SentenceTokenizer, extend_vocab, coverage
from deepmoji.tokenizer import tokenize

DATASETS = [
    'Olympic',
    'PsychExp',
    'SCv1',
    'SCv2-GEN',
    'SE0714',
    # 'SE1604', # Excluded due to Twitter's ToS
    'SS-Twitter',
    'SS-Youtube',
]

DIR = '../data'
FILENAME_RAW = 'raw.pickle'
FILENAME_OWN = 'own_vocab.pickle'
FILENAME_OUR = 'twitter_vocab.pickle'
FILENAME_COMBINED = 'combined_vocab.pickle'


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def format_pickle(dset, train_texts, val_texts, test_texts, train_labels, val_labels, test_labels):
    return {'dataset': dset,
            'train_texts': train_texts,
            'val_texts': val_texts,
            'test_texts': test_texts,
            'train_labels': train_labels,
            'val_labels': val_labels,
            'test_labels': test_labels}


def convert_dataset(filepath, extend_with, vocab):
    print('-- Generating {} '.format(filepath))
    sys.stdout.flush()
    st = SentenceTokenizer(vocab, maxlen)
    tokenized, dicts, _ = st.split_train_val_test(texts,
                                                  labels,
                                                  [data['train_ind'],
                                                   data['val_ind'],
                                                   data['test_ind']],
                                                  extend_with=extend_with)
    pick = format_pickle(dset, tokenized[0], tokenized[1], tokenized[2],
                         dicts[0], dicts[1], dicts[2])
    with open(filepath, 'w') as f:
        pickle.dump(pick, f)
    cover = coverage(tokenized[2])

    print('     done. Coverage: {}'.format(cover))


with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

for dset in DATASETS:
    print('Converting {}'.format(dset))

    PATH_RAW = '{}/{}/{}'.format(DIR, dset, FILENAME_RAW)
    PATH_OWN = '{}/{}/{}'.format(DIR, dset, FILENAME_OWN)
    PATH_OUR = '{}/{}/{}'.format(DIR, dset, FILENAME_OUR)
    PATH_COMBINED = '{}/{}/{}'.format(DIR, dset, FILENAME_COMBINED)

    with open(PATH_RAW) as dataset:
        data = pickle.load(dataset)

    # Decode data
    try:
        texts = [unicode(x) for x in data['texts']]
    except UnicodeDecodeError:
        texts = [x.decode('utf-8') for x in data['texts']]

    wg = WordGenerator(texts)
    vb = VocabBuilder(wg)
    vb.count_all_words()

    # Calculate max length of sequences considered
    # Adjust batch_size accordingly to prevent GPU overflow
    lengths = [len(tokenize(t)) for t in texts]
    maxlen = roundup(np.percentile(lengths, 80.0))

    # Extract labels
    labels = [x['label'] for x in data['info']]

    convert_dataset(PATH_OWN, 50000, {})
    convert_dataset(PATH_OUR, 0, vocab)
    convert_dataset(PATH_COMBINED, 10000, vocab)
