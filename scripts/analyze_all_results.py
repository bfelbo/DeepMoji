from __future__ import print_function

# allow us to import the codebase/keras directory
import sys
import glob
import numpy as np
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

DATASETS = ['SE0714', 'Olympic', 'PsychExp', 'SS-Twitter', 'SS-Youtube',
            'SCv1', 'SV2-GEN']  # 'SE1604' excluded due to Twitter's ToS


def get_results(dset):
    METHOD = 'last'
    RESULTS_DIR = 'results/'
    RESULT_PATHS = glob.glob('{}/{}_{}_*_results.txt'.format(RESULTS_DIR, dset, METHOD))
    assert len(RESULT_PATHS)

    scores = []
    for path in RESULT_PATHS:
        with open(path) as f:
            score = f.readline().split(':')[1]
        scores.append(float(score))

    average = np.mean(scores)
    maximum = max(scores)
    minimum = min(scores)
    std = np.std(scores)

    print('Dataset: {}'.format(dset))
    print('Method:  {}'.format(METHOD))
    print('Number of results: {}'.format(len(scores)))
    print('--------------------------')
    print('Average: {}'.format(average))
    print('Maximum: {}'.format(maximum))
    print('Minimum: {}'.format(minimum))
    print('Standard deviaton: {}'.format(std))


for dset in DATASETS:
    get_results(dset)
