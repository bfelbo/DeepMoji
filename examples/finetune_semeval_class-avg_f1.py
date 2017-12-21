"""Finetuning example.

Trains the DeepMoji model on the SemEval emotion dataset, using the 'last'
finetuning method and the class average F1 metric.

The 'last' method does the following:
0) Load all weights except for the softmax layer. Do not add tokens to the
   vocabulary and do not extend the embedding layer.
1) Freeze all layers except for the softmax layer.
2) Train.

The class average F1 metric does the following:
1) For each class, relabel the dataset into binary classification
   (belongs to/does not belong to this class).
2) Calculate F1 score for each class.
3) Compute the average of all F1 scores.
"""

from __future__ import print_function
import example_helper
import json
from deepmoji.finetuning import load_benchmark
from deepmoji.class_avg_finetuning import class_avg_finetune
from deepmoji.model_def import deepmoji_transfer
from deepmoji.global_variables import PRETRAINED_PATH

DATASET_PATH = '../data/SE0714/raw.pickle'
nb_classes = 3

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)


# Load dataset. Extend the existing vocabulary with up to 10000 tokens from
# the training dataset.
data = load_benchmark(DATASET_PATH, vocab, extend_with=10000)

# Set up model and finetune. Note that we have to extend the embedding layer
# with the number of tokens added to the vocabulary.
#
# Also note that when using class average F1 to evaluate, the model has to be
# defined with two classes, since the model will be trained for each class
# separately.
model = deepmoji_transfer(2, data['maxlen'], PRETRAINED_PATH,
                          extend_embedding=data['added'])
model.summary()

# For finetuning however, pass in the actual number of classes.
model, f1 = class_avg_finetune(model, data['texts'], data['labels'],
                               nb_classes, data['batch_size'], method='last')
print('F1: {}'.format(f1))
