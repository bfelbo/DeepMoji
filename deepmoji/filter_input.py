
import codecs
import csv
import numpy as np
from emoji import UNICODE_EMOJI


def read_english(path="english_words.txt", add_emojis=True):
    # read english words for filtering (includes emojis as part of set)
    english = set()
    with codecs.open(path, "r", "utf-8") as f:
        for line in f:
            line = line.strip().lower().replace('\n', '')
            if len(line):
                english.add(line)
    if add_emojis:
        for e in UNICODE_EMOJI:
            english.add(e)
    return english


def read_wanted_emojis(path="wanted_emojis.csv"):
    emojis = []
    with open(path, 'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            line = line[0].strip().replace('\n', '')
            line = line.decode('unicode-escape')
            emojis.append(line)
    return emojis


def read_non_english_users(path="unwanted_users.npz"):
    try:
        neu_set = set(np.load(path)['userids'])
    except IOError:
        neu_set = set()
    return neu_set
