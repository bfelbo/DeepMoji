""" Creates a vocabulary from a tsv file.
"""

import codecs
import example_helper
from deepmoji.create_vocab import VocabBuilder
from deepmoji.word_generator import TweetWordGenerator

with codecs.open('../../twitterdata/tweets.2016-09-01', 'rU', 'utf-8') as stream:
    wg = TweetWordGenerator(stream)
    vb = VocabBuilder(wg)
    vb.count_all_words()
    vb.save_vocab()
