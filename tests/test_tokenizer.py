# -*- coding: utf-8 -*-
""" Tokenization tests.
"""
import sys
from nose.tools import nottest
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from deepmoji.tokenizer import tokenize

TESTS_NORMAL = [
    (u'200K words!', [u'200', u'K', u'words', u'!']),
]

TESTS_EMOJIS = [
    (u'i \U0001f496 you to the moon and back',
     [u'i', u'\U0001f496', u'you', u'to', u'the', u'moon', u'and', u'back']),
    (u"i\U0001f496you to the \u2605's and back",
     [u'i', u'\U0001f496', u'you', u'to', u'the',
      u'\u2605', u"'", u's', u'and', u'back']),
    (u'~<3~', [u'~', u'<3', u'~']),
    (u'<333', [u'<333']),
    (u':-)', [u':-)']),
    (u'>:-(', [u'>:-(']),
    (u'\u266b\u266a\u2605\u2606\u2665\u2764\u2661',
     [u'\u266b', u'\u266a', u'\u2605', u'\u2606',
      u'\u2665', u'\u2764', u'\u2661']),
]

TESTS_URLS = [
    (u'www.sample.com', [u'www.sample.com']),
    (u'http://endless.horse', [u'http://endless.horse']),
    (u'https://github.mit.edu', [u'https://github.mit.edu']),
]

TESTS_TWITTER = [
    (u'#blacklivesmatter', [u'#blacklivesmatter']),
    (u'#99_percent.', [u'#99_percent', u'.']),
    (u'the#99%', [u'the', u'#99', u'%']),
    (u'@golden_zenith', [u'@golden_zenith']),
    (u'@99_percent', [u'@99_percent']),
    (u'latte-express@mit.edu', [u'latte-express@mit.edu']),
]

TESTS_PHONE_NUMS = [
    (u'518)528-0252', [u'518', u')', u'528', u'-', u'0252']),
    (u'1200-0221-0234', [u'1200', u'-', u'0221', u'-', u'0234']),
    (u'1200.0221.0234', [u'1200', u'.', u'0221', u'.', u'0234']),
]

TESTS_DATETIME = [
    (u'15:00', [u'15', u':', u'00']),
    (u'2:00pm', [u'2', u':', u'00', u'pm']),
    (u'9/14/16', [u'9', u'/', u'14', u'/', u'16']),
]

TESTS_CURRENCIES = [
    (u'517.933\xa3', [u'517', u'.', u'933', u'\xa3']),
    (u'$517.87', [u'$', u'517', u'.', u'87']),
    (u'1201.6598', [u'1201', u'.', u'6598']),
    (u'120,6', [u'120', u',', u'6']),
    (u'10,00\u20ac', [u'10', u',', u'00', u'\u20ac']),
    (u'1,000', [u'1', u',', u'000']),
    (u'1200pesos', [u'1200', u'pesos']),
]

TESTS_NUM_SYM = [
    (u'5162f', [u'5162', u'f']),
    (u'f5162', [u'f', u'5162']),
    (u'1203(', [u'1203', u'(']),
    (u'(1203)', [u'(', u'1203', u')']),
    (u'1200/', [u'1200', u'/']),
    (u'1200+', [u'1200', u'+']),
    (u'1202o-east', [u'1202', u'o-east']),
    (u'1200r', [u'1200', u'r']),
    (u'1200-1400', [u'1200', u'-', u'1400']),
    (u'120/today', [u'120', u'/', u'today']),
    (u'today/120', [u'today', u'/', u'120']),
    (u'120/5', [u'120', u'/', u'5']),
    (u"120'/5", [u'120', u"'", u'/', u'5']),
    (u'120/5pro', [u'120', u'/', u'5', u'pro']),
    (u"1200's,)", [u'1200', u"'", u's', u',', u')']),
    (u'120.76.218.207', [u'120', u'.', u'76', u'.', u'218', u'.', u'207']),
]

TESTS_PUNCTUATION = [
    (u"don''t", [u'don', u"''", u't']),
    (u"don'tcha", [u"don'tcha"]),
    (u'no?!?!;', [u'no', u'?', u'!', u'?', u'!', u';']),
    (u'no??!!..', [u'no', u'??', u'!!', u'..']),
    (u'a.m.', [u'a.m.']),
    (u'.s.u', [u'.', u's', u'.', u'u']),
    (u'!!i..n__', [u'!!', u'i', u'..', u'n', u'__']),
    (u'lv(<3)w(3>)u Mr.!', [u'lv', u'(', u'<3', u')', u'w', u'(', u'3',
                            u'>', u')', u'u', u'Mr.', u'!']),
    (u'-->', [u'--', u'>']),
    (u'->', [u'-', u'>']),
    (u'<-', [u'<', u'-']),
    (u'<--', [u'<', u'--']),
    (u'hello (@person)', [u'hello', u'(', u'@person', u')']),
]


def test_normal():
    """ Normal/combined usage.
    """
    test_base(TESTS_NORMAL)


def test_emojis():
    """ Tokenizing emojis/emoticons/decorations.
    """
    test_base(TESTS_EMOJIS)


def test_urls():
    """ Tokenizing URLs.
    """
    test_base(TESTS_URLS)


def test_twitter():
    """ Tokenizing hashtags, mentions and emails.
    """
    test_base(TESTS_TWITTER)


def test_phone_nums():
    """ Tokenizing phone numbers.
    """
    test_base(TESTS_PHONE_NUMS)


def test_datetime():
    """ Tokenizing dates and times.
    """
    test_base(TESTS_DATETIME)


def test_currencies():
    """ Tokenizing currencies.
    """
    test_base(TESTS_CURRENCIES)


def test_num_sym():
    """ Tokenizing combinations of numbers and symbols.
    """
    test_base(TESTS_NUM_SYM)


def test_punctuation():
    """ Tokenizing punctuation and contractions.
    """
    test_base(TESTS_PUNCTUATION)


@nottest
def test_base(tests):
    """ Base function for running tests.
    """
    for (test, expected) in tests:
        actual = tokenize(test)
        assert actual == expected, \
            "Tokenization of \'{}\' failed, expected: {}, actual: {}"\
            .format(test, expected, actual)
