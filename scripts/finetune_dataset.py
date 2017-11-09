""" Finetuning example.
"""
from __future__ import print_function
import sys
import numpy as np
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import json
import math
from deepmoji.model_def import deepmoji_transfer
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from deepmoji.finetuning import (
    load_benchmark,
    finetune)
from deepmoji.class_avg_finetuning import class_avg_finetune


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


# Format: (dataset_name,
#          path_to_dataset,
#          nb_classes,
#          use_f1_score)
DATASETS = [
    # ('SE0714', '../data/SE0714/raw.pickle', 3, True),
    # ('Olympic', '../data/Olympic/raw.pickle', 4, True),
    # ('PsychExp', '../data/PsychExp/raw.pickle', 7, True),
    # ('SS-Twitter', '../data/SS-Twitter/raw.pickle', 2, False),
    ('SS-Youtube', '../data/SS-Youtube/raw.pickle', 2, False),
    # ('SE1604', '../data/SE1604/raw.pickle', 3, False), # Excluded due to Twitter's ToS
    # ('SCv1', '../data/SCv1/raw.pickle', 2, True),
    # ('SCv2-GEN', '../data/SCv2-GEN/raw.pickle', 2, True)
]

RESULTS_DIR = 'results'

# 'new' | 'last' | 'full' | 'chain-thaw'
FINETUNE_METHOD = 'last'
VERBOSE = 1

nb_tokens = 50000
nb_epochs = 1000
epoch_size = 1000

with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

for rerun_iter in range(5):
    for p in DATASETS:

        # debugging
        assert len(vocab) == nb_tokens

        dset = p[0]
        path = p[1]
        nb_classes = p[2]
        use_f1_score = p[3]

        if FINETUNE_METHOD == 'last':
            extend_with = 0
        elif FINETUNE_METHOD in ['new', 'full', 'chain-thaw']:
            extend_with = 10000
        else:
            raise ValueError('Finetuning method not recognised!')

        # Load dataset.
        data = load_benchmark(path, vocab, extend_with=extend_with)

        (X_train, y_train) = (data['texts'][0], data['labels'][0])
        (X_val, y_val) = (data['texts'][1], data['labels'][1])
        (X_test, y_test) = (data['texts'][2], data['labels'][2])

        weight_path = PRETRAINED_PATH if FINETUNE_METHOD != 'new' else None
        nb_model_classes = 2 if use_f1_score else nb_classes
        model = deepmoji_transfer(
            nb_model_classes,
            data['maxlen'], weight_path,
            extend_embedding=data['added'])
        model.summary()

        # Training
        print('Training: {}'.format(path))
        if use_f1_score:
            model, result = class_avg_finetune(model, data['texts'],
                                               data['labels'],
                                               nb_classes, data['batch_size'],
                                               FINETUNE_METHOD,
                                               verbose=VERBOSE)
        else:
            model, result = finetune(model, data['texts'], data['labels'],
                                     nb_classes, data['batch_size'],
                                     FINETUNE_METHOD, metric='acc',
                                     verbose=VERBOSE)

        # Write results
        if use_f1_score:
            print('Overall F1 score (dset = {}): {}'.format(dset, result))
            with open('{}/{}_{}_{}_results.txt'.
                      format(RESULTS_DIR, dset, FINETUNE_METHOD, rerun_iter),
                      "w") as f:
                f.write("F1: {}\n".format(result))
        else:
            print('Test accuracy (dset = {}): {}'.format(dset, result))
            with open('{}/{}_{}_{}_results.txt'.
                      format(RESULTS_DIR, dset, FINETUNE_METHOD, rerun_iter),
                      "w") as f:
                f.write("Acc: {}\n".format(result))
