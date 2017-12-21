"""
Take a given list of sentences and turn it into a numpy array, where each
number corresponds to a word. Padding is used (number 0) to ensure fixed length
of sentences.
"""

from __future__ import print_function
import example_helper
import json
from deepmoji.sentence_tokenizer import SentenceTokenizer

with open('../model/vocabulary.json', 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, 30)
test_sentences = [
    u'\u2014 -- \u203c !!\U0001F602',
    u'Hello world!',
    u'This is a sample tweet #example',
]

tokens, infos, stats = st.tokenize_sentences(test_sentences)

print(tokens)
print(infos)
print(stats)
