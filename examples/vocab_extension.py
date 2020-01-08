"""
Extend the given vocabulary using dataset-specific words.

1. First create a vocabulary for the specific dataset.
2. Find all words not in our vocabulary, but in the dataset vocabulary.
3. Take top X (default=1000) of these words and add them to the vocabulary.
4. Save this combined vocabulary and embedding matrix, which can now be used.
"""


from . import example_helper
import json
from deepmoji.create_vocab import extend_vocab, VocabBuilder
from deepmoji.word_generator import WordGenerator

new_words = ['#zzzzaaazzz', 'newword', 'newword']
word_gen = WordGenerator(new_words)
vb = VocabBuilder(word_gen)
vb.count_all_words()

with open('../model/vocabulary.json') as f:
    vocab = json.load(f)

print(len(vocab))
print(vb.word_counts)
extend_vocab(vocab, vb, max_tokens=1)

# 'newword' should be added because it's more frequent in the given vocab
print(vocab['newword'])
print(len(vocab))
