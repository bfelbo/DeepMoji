# -*- coding: utf-8 -*-
""" Tokenization tests.
"""

from nose.tools import nottest

from deepmoji.tokenizer import tokenize

TESTS_NORMAL = [
    ('200K words!', ['200', 'K', 'words', '!']),
]

TESTS_EMOJIS = [
    ('i \U0001f496 you to the moon and back',
     ['i', '\U0001f496', 'you', 'to', 'the', 'moon', 'and', 'back']),
    ("i\U0001f496you to the \u2605's and back",
     ['i', '\U0001f496', 'you', 'to', 'the',
      '\u2605', "'", 's', 'and', 'back']),
    ('~<3~', ['~', '<3', '~']),
    ('<333', ['<333']),
    (':-)', [':-)']),
    ('>:-(', ['>:-(']),
    ('\u266b\u266a\u2605\u2606\u2665\u2764\u2661',
     ['\u266b', '\u266a', '\u2605', '\u2606',
      '\u2665', '\u2764', '\u2661']),
]

TESTS_URLS = [
    ('www.sample.com', ['www.sample.com']),
    ('http://endless.horse', ['http://endless.horse']),
    ('https://github.mit.edu', ['https://github.mit.edu']),
]

TESTS_TWITTER = [
    ('#blacklivesmatter', ['#blacklivesmatter']),
    ('#99_percent.', ['#99_percent', '.']),
    ('the#99%', ['the', '#99', '%']),
    ('@golden_zenith', ['@golden_zenith']),
    ('@99_percent', ['@99_percent']),
    ('latte-express@mit.edu', ['latte-express@mit.edu']),
]

TESTS_PHONE_NUMS = [
    ('518)528-0252', ['518', ')', '528', '-', '0252']),
    ('1200-0221-0234', ['1200', '-', '0221', '-', '0234']),
    ('1200.0221.0234', ['1200', '.', '0221', '.', '0234']),
]

TESTS_DATETIME = [
    ('15:00', ['15', ':', '00']),
    ('2:00pm', ['2', ':', '00', 'pm']),
    ('9/14/16', ['9', '/', '14', '/', '16']),
]

TESTS_CURRENCIES = [
    ('517.933\xa3', ['517', '.', '933', '\xa3']),
    ('$517.87', ['$', '517', '.', '87']),
    ('1201.6598', ['1201', '.', '6598']),
    ('120,6', ['120', ',', '6']),
    ('10,00\u20ac', ['10', ',', '00', '\u20ac']),
    ('1,000', ['1', ',', '000']),
    ('1200pesos', ['1200', 'pesos']),
]

TESTS_NUM_SYM = [
    ('5162f', ['5162', 'f']),
    ('f5162', ['f', '5162']),
    ('1203(', ['1203', '(']),
    ('(1203)', ['(', '1203', ')']),
    ('1200/', ['1200', '/']),
    ('1200+', ['1200', '+']),
    ('1202o-east', ['1202', 'o-east']),
    ('1200r', ['1200', 'r']),
    ('1200-1400', ['1200', '-', '1400']),
    ('120/today', ['120', '/', 'today']),
    ('today/120', ['today', '/', '120']),
    ('120/5', ['120', '/', '5']),
    ("120'/5", ['120', "'", '/', '5']),
    ('120/5pro', ['120', '/', '5', 'pro']),
    ("1200's,)", ['1200', "'", 's', ',', ')']),
    ('120.76.218.207', ['120', '.', '76', '.', '218', '.', '207']),
]

TESTS_PUNCTUATION = [
    ("don''t", ['don', "''", 't']),
    ("don'tcha", ["don'tcha"]),
    ('no?!?!;', ['no', '?', '!', '?', '!', ';']),
    ('no??!!..', ['no', '??', '!!', '..']),
    ('a.m.', ['a.m.']),
    ('.s.u', ['.', 's', '.', 'u']),
    ('!!i..n__', ['!!', 'i', '..', 'n', '__']),
    ('lv(<3)w(3>)u Mr.!', ['lv', '(', '<3', ')', 'w', '(', '3',
                            '>', ')', 'u', 'Mr.', '!']),
    ('-->', ['--', '>']),
    ('->', ['-', '>']),
    ('<-', ['<', '-']),
    ('<--', ['<', '--']),
    ('hello (@person)', ['hello', '(', '@person', ')']),
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
