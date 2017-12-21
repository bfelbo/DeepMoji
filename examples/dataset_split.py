'''
Split a given dataset into three different datasets: training, validation and
testing.

This is achieved by splitting the given list of sentences into three separate
lists according to either a given ratio (e.g. [0.7, 0.1, 0.2]) or by an
explicit enumeration. The sentences are also tokenised using the given
vocabulary.

Also splits a given list of dictionaries containing information about
each sentence.

An additional parameter can be set 'extend_with', which will extend the given
vocabulary with up to 'extend_with' tokens, taken from the training dataset.
'''
from __future__ import print_function
import example_helper
import json

from deepmoji.sentence_tokenizer import SentenceTokenizer

DATASET = [
    u'I am sentence 0',
    u'I am sentence 1',
    u'I am sentence 2',
    u'I am sentence 3',
    u'I am sentence 4',
    u'I am sentence 5',
    u'I am sentence 6',
    u'I am sentence 7',
    u'I am sentence 8',
    u'I am sentence 9 newword',
]

INFO_DICTS = [
    {'label': 'sentence 0'},
    {'label': 'sentence 1'},
    {'label': 'sentence 2'},
    {'label': 'sentence 3'},
    {'label': 'sentence 4'},
    {'label': 'sentence 5'},
    {'label': 'sentence 6'},
    {'label': 'sentence 7'},
    {'label': 'sentence 8'},
    {'label': 'sentence 9'},
]

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)
st = SentenceTokenizer(vocab, 30)

# Split using the default split ratio
print(st.split_train_val_test(DATASET, INFO_DICTS))

# Split explicitly
print(st.split_train_val_test(DATASET,
                              INFO_DICTS,
                              [[0, 1, 2, 4, 9], [5, 6], [7, 8, 3]],
                              extend_with=1))
