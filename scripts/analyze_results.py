from __future__ import print_function

import sys
import glob
import numpy as np

DATASET = 'SS-Twitter' # 'SE1604' excluded due to Twitter's ToS
METHOD = 'new'

# Optional usage: analyze_results.py <dataset> <method>
if len(sys.argv) == 3:
    DATASET = sys.argv[1]
    METHOD = sys.argv[2]

RESULTS_DIR = 'results/'
RESULT_PATHS = glob.glob('{}/{}_{}_*_results.txt'.format(RESULTS_DIR, DATASET, METHOD))

if not RESULT_PATHS:
    print('Could not find results for \'{}\' using \'{}\' in directory \'{}\'.'.format(DATASET, METHOD, RESULTS_DIR))
else:
    scores = []
    for path in RESULT_PATHS:
        with open(path) as f:
            score = f.readline().split(':')[1]
        scores.append(float(score))

    average = np.mean(scores)
    maximum = max(scores)
    minimum = min(scores)
    std = np.std(scores)

    print('Dataset: {}'.format(DATASET))
    print('Method:  {}'.format(METHOD))
    print('Number of results: {}'.format(len(scores)))
    print('--------------------------')
    print('Average: {}'.format(average))
    print('Maximum: {}'.format(maximum))
    print('Minimum: {}'.format(minimum))
    print('Standard deviaton: {}'.format(std))
