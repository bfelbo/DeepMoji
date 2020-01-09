
import os
from hashlib import sha256
from os.path import dirname, abspath, join

import requests

WEIGHTS_FILENAME = "deepmoji_weights.hdf5"
MODEL_DIR = join(dirname(dirname(abspath(__file__))), 'model')
WEIGHTS_PATH = join(MODEL_DIR, WEIGHTS_FILENAME)

WEIGHTS_DOWNLOAD_LINK = 'https://dl.dropboxusercontent.com/s/xqarafsl6a8f9ny/deepmoji_weights.hdf5'
WEIGHTS_FILE_SHA_256 = "ca663315cf4a22ced569cf928f9277ebd013c55d93a19e1d0dde001f59a72476"


def prompt():
    while True:
        valid = {
            'y': True,
            'ye': True,
            'yes': True,
            'n': False,
            'no': False,
        }
        if 'TRAVIS' in os.environ:
            choice = 'yes'
        else:
            choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            print('Please respond with \'y\' or \'n\' (or \'yes\' or \'no\')')


download = True
if os.path.exists(WEIGHTS_PATH):
    print('Weight file already exists at {}. Would you like to redownload it anyway? [y/n]'.format(WEIGHTS_PATH))
    download = prompt()
    already_exists = True
else:
    already_exists = False

if download:
    print('About to download the pretrained weights file from: {}'.format(WEIGHTS_DOWNLOAD_LINK))
    if not already_exists:
        print('The size of the file is roughly 85MB. Continue? [y/n]')
    else:
        os.unlink(WEIGHTS_PATH)

    if already_exists or prompt():
        print('Downloading...')

        with open(WEIGHTS_PATH, 'wb') as f:
            f.write(requests.get(WEIGHTS_DOWNLOAD_LINK).content)

        resp = requests.get(WEIGHTS_DOWNLOAD_LINK)
        m = sha256()
        m.update(resp.content)
        if (m.hexdigest() != WEIGHTS_FILE_SHA_256):
            raise ValueError("Downloaded weights sha256sum: {} is not the expected: {}".format(m.hexdigest(), WEIGHTS_FILE_SHA_256))
        with open(os.path.abspath(WEIGHTS_PATH), "wb") as f:
            f.write(resp.content)
        print('Downloaded weights to: {}'.format(WEIGHTS_PATH))
