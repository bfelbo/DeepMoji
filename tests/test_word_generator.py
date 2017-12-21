# -*- coding: utf-8 -*-
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from nose.tools import raises
from deepmoji.word_generator import WordGenerator


@raises(ValueError)
def test_only_unicode_accepted():
    """ Non-Unicode strings raise a ValueError.
    """
    sentences = [
        u'Hello world',
        u'I am unicode',
        'I am not unicode',
    ]

    wg = WordGenerator(sentences)
    for w in wg:
        pass


def test_unicode_sentences_ignored_if_set():
    """ Strings with Unicode characters tokenize to empty array if they're not allowed.
    """
    sentence = [u'Dobrý den, jak se máš?']
    wg = WordGenerator(sentence, allow_unicode_text=False)
    assert wg.get_words(sentence[0]) == []


def test_check_ascii():
    """ check_ascii recognises ASCII words properly.
    """
    wg = WordGenerator([])
    assert wg.check_ascii('ASCII')
    assert not wg.check_ascii('ščřžýá')
    assert not wg.check_ascii('❤ ☀ ☆ ☂ ☻ ♞ ☯ ☭ ☢')


def test_convert_unicode_word():
    """ convert_unicode_word converts Unicode words correctly.
    """
    wg = WordGenerator([], allow_unicode_text=True)

    result = wg.convert_unicode_word(u'č')
    assert result == (True, u'\u010d'), '{}'.format(result)


def test_convert_unicode_word_ignores_if_set():
    """ convert_unicode_word ignores Unicode words if set.
    """
    wg = WordGenerator([], allow_unicode_text=False)

    result = wg.convert_unicode_word(u'č')
    assert result == (False, ''), '{}'.format(result)


def test_convert_unicode_chars():
    """ convert_unicode_word correctly converts accented characters.
    """
    wg = WordGenerator([], allow_unicode_text=True)
    result = wg.convert_unicode_word(u'ěščřžýáíé')
    assert result == (True, u'\u011b\u0161\u010d\u0159\u017e\xfd\xe1\xed\xe9'), '{}'.format(result)
