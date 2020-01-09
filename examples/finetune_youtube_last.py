"""Finetuning example.

Trains the DeepMoji model on the SS-Youtube dataset, using the 'last'
finetuning method and the accuracy metric.

The 'last' method does the following:
0) Load all weights except for the softmax layer. Do not add tokens to the
   vocabulary and do not extend the embedding layer.
1) Freeze all layers except for the softmax layer.
2) Train.
"""

import json

from deepmoji.finetuning import (
    load_benchmark,
    finetune)
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from deepmoji.model_def import deepmoji_transfer

DATASET_PATH = '../data/SS-Youtube/raw.pickle'
nb_classes = 2

with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

# Load dataset.
data = load_benchmark(DATASET_PATH, vocab)

# Set up model and finetune
model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH)
model.summary()
model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                      data['batch_size'], method='last')
print('Acc: {}'.format(acc))
