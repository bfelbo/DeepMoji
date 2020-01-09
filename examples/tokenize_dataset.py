"""
Take a given list of sentences and turn it into a numpy array, where each
number corresponds to a word. Padding is used (number 0) to ensure fixed length
of sentences.
"""

import json

from deepmoji.global_variables import VOCAB_PATH
from deepmoji.sentence_tokenizer import SentenceTokenizer

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, 30)
test_sentences = [
    '\u2014 -- \u203c !!\U0001F602',
    'Hello world!',
    'This is a sample tweet #example',
]

tokens, infos, stats = st.tokenize_sentences(test_sentences)

print(tokens)
print(infos)
print(stats)
