
from __future__ import print_function, division
import sys
import numpy as np
import re
import string
import emoji
from tokenizer import RE_MENTION, RE_URL
from global_variables import SPECIAL_TOKENS
from itertools import groupby

AtMentionRegex = re.compile(RE_MENTION)
urlRegex = re.compile(RE_URL)

# from http://bit.ly/2rdjgjE (UTF-8 encodings and Unicode chars)
VARIATION_SELECTORS = [u'\ufe00',
                       u'\ufe01',
                       u'\ufe02',
                       u'\ufe03',
                       u'\ufe04',
                       u'\ufe05',
                       u'\ufe06',
                       u'\ufe07',
                       u'\ufe08',
                       u'\ufe09',
                       u'\ufe0a',
                       u'\ufe0b',
                       u'\ufe0c',
                       u'\ufe0d',
                       u'\ufe0e',
                       u'\ufe0f']

# from https://stackoverflow.com/questions/92438/stripping-non-printable-characters-from-a-string-in-python
ALL_CHARS = (unichr(i) for i in xrange(sys.maxunicode))
CONTROL_CHARS = ''.join(map(unichr, range(0, 32) + range(127, 160)))
CONTROL_CHAR_REGEX = re.compile('[%s]' % re.escape(CONTROL_CHARS))


def is_special_token(word):
    equal = False
    for spec in SPECIAL_TOKENS:
        if word == spec:
            equal = True
            break
    return equal


def mostly_english(words, english, pct_eng_short=0.5, pct_eng_long=0.6, ignore_special_tokens=True, min_length=2):
    """ Ensure text meets threshold for containing English words """

    n_words = 0
    n_english = 0

    if english is None:
        return True, 0, 0

    for w in words:
        if len(w) < min_length:
            continue
        if punct_word(w):
            continue
        if ignore_special_tokens and is_special_token(w):
            continue
        n_words += 1
        if w in english:
            n_english += 1

    if n_words < 2:
        return True, n_words, n_english
    if n_words < 5:
        valid_english = n_english >= n_words * pct_eng_short
    else:
        valid_english = n_english >= n_words * pct_eng_long
    return valid_english, n_words, n_english


def correct_length(words, min_words, max_words, ignore_special_tokens=True):
    """ Ensure text meets threshold for containing English words
        and that it's within the min and max words limits. """

    if min_words is None:
        min_words = 0

    if max_words is None:
        max_words = 99999

    n_words = 0
    for w in words:
        if punct_word(w):
            continue
        if ignore_special_tokens and is_special_token(w):
            continue
        n_words += 1
    valid = min_words <= n_words and n_words <= max_words
    return valid


def punct_word(word, punctuation=string.punctuation):
    return all([True if c in punctuation else False for c in word])


def load_non_english_user_set():
    non_english_user_set = set(np.load('uids.npz')['data'])
    return non_english_user_set


def non_english_user(userid, non_english_user_set):
    neu_found = int(userid) in non_english_user_set
    return neu_found


def separate_emojis_and_text(text):
    emoji_chars = []
    non_emoji_chars = []
    for c in text:
        if c in emoji.UNICODE_EMOJI:
            emoji_chars.append(c)
        else:
            non_emoji_chars.append(c)
    return ''.join(emoji_chars), ''.join(non_emoji_chars)


def extract_emojis(text, wanted_emojis):
    text = remove_variation_selectors(text)
    return [c for c in text if c in wanted_emojis]


def remove_variation_selectors(text):
    """ Remove styling glyph variants for Unicode characters.
        For instance, remove skin color from emojis.
    """
    for var in VARIATION_SELECTORS:
        text = text.replace(var, u'')
    return text


def shorten_word(word):
    """ Shorten groupings of 3+ identical consecutive chars to 2, e.g. '!!!!' --> '!!'
    """

    # only shorten ASCII words
    try:
        word.decode('ascii')
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        return word

    # must have at least 3 char to be shortened
    if len(word) < 3:
        return word

    # find groups of 3+ consecutive letters
    letter_groups = [list(g) for k, g in groupby(word)]
    triple_or_more = [''.join(g) for g in letter_groups if len(g) >= 3]
    if len(triple_or_more) == 0:
        return word

    # replace letters to find the short word
    short_word = word
    for trip in triple_or_more:
        short_word = short_word.replace(trip, trip[0] * 2)

    return short_word


def detect_special_tokens(word):
    try:
        int(word)
        word = SPECIAL_TOKENS[4]
    except ValueError:
        if AtMentionRegex.findall(word):
            word = SPECIAL_TOKENS[2]
        elif urlRegex.findall(word):
            word = SPECIAL_TOKENS[3]
    return word


def process_word(word):
    """ Shortening and converting the word to a special token if relevant.
    """
    word = shorten_word(word)
    word = detect_special_tokens(word)
    return word


def remove_control_chars(text):
    return CONTROL_CHAR_REGEX.sub('', text)


def convert_nonbreaking_space(text):
    # ugly hack handling non-breaking space no matter how badly it's been encoded in the input
    for r in [u'\\\\xc2', u'\\xc2', u'\xc2', u'\\\\xa0', u'\\xa0', u'\xa0']:
        text = text.replace(r, u' ')
    return text


def convert_linebreaks(text):
    # ugly hack handling non-breaking space no matter how badly it's been encoded in the input
    # space around to ensure proper tokenization
    for r in [u'\\\\n', u'\\n', u'\n', u'\\\\r', u'\\r', u'\r', '<br>']:
        text = text.replace(r, u' ' + SPECIAL_TOKENS[5] + u' ')
    return text
