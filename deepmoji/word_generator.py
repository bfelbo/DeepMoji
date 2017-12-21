''' Extracts lists of words from a given input to be used for later vocabulary
    generation or for creating tokenized datasets.
    Supports functionality for handling different file types and
    filtering/processing of this input.
'''

from __future__ import division, print_function

import re
import unicodedata
import numpy as np
from text_unidecode import unidecode
from tokenizer import RE_MENTION, tokenize
from filter_utils import (
    convert_linebreaks,
    convert_nonbreaking_space,
    correct_length,
    extract_emojis,
    mostly_english,
    non_english_user,
    process_word,
    punct_word,
    remove_control_chars,
    remove_variation_selectors,
    separate_emojis_and_text)

# Only catch retweets in the beginning of the tweet as those are the
# automatically added ones.
# We do not want to remove tweets like "Omg.. please RT this!!"
RETWEETS_RE = re.compile(r'^[rR][tT]')

# Use fast and less precise regex for removing tweets with URLs
# It doesn't matter too much if a few tweets with URL's make it through
URLS_RE = re.compile(r'https?://|www\.')

MENTION_RE = re.compile(RE_MENTION)
ALLOWED_CONVERTED_UNICODE_PUNCTUATION = """!"#$'()+,-.:;<=>?@`~"""


class WordGenerator():
    ''' Cleanses input and converts into words. Needs all sentences to be in
        Unicode format. Has subclasses that read sentences differently based on
        file type.

    Takes a generator as input. This can be from e.g. a file.
    unicode_handling in ['ignore_sentence', 'convert_punctuation', 'allow']
    unicode_handling in ['ignore_emoji', 'ignore_sentence', 'allow']
    '''

    def __init__(self, stream, allow_unicode_text=False, ignore_emojis=True,
                 remove_variation_selectors=True, break_replacement=True):
        self.stream = stream
        self.allow_unicode_text = allow_unicode_text
        self.remove_variation_selectors = remove_variation_selectors
        self.ignore_emojis = ignore_emojis
        self.break_replacement = break_replacement
        self.reset_stats()

    def get_words(self, sentence):
        """ Tokenizes a sentence into individual words.
            Converts Unicode punctuation into ASCII if that option is set.
            Ignores sentences with Unicode if that option is set.
            Returns an empty list of words if the sentence has Unicode and
            that is not allowed.
        """

        if not isinstance(sentence, unicode):
            raise ValueError("All sentences should be Unicode-encoded!")
        sentence = sentence.strip().lower()

        if self.break_replacement:
            sentence = convert_linebreaks(sentence)

        if self.remove_variation_selectors:
            sentence = remove_variation_selectors(sentence)

        # Split into words using simple whitespace splitting and convert
        # Unicode. This is done to prevent word splitting issues with
        # twokenize and Unicode
        words = sentence.split()
        converted_words = []
        for w in words:
            accept_sentence, c_w = self.convert_unicode_word(w)
            # Unicode word detected and not allowed
            if not accept_sentence:
                return []
            else:
                converted_words.append(c_w)
        sentence = ' '.join(converted_words)

        words = tokenize(sentence)
        words = [process_word(w) for w in words]
        return words

    def check_ascii(self, word):
        """ Returns whether a word is ASCII """

        try:
            word.decode('ascii')
            return True
        except (UnicodeDecodeError, UnicodeEncodeError):
            return False

    def convert_unicode_punctuation(self, word):
        word_converted_punct = []
        for c in word:
            decoded_c = unidecode(c).lower()
            if len(decoded_c) == 0:
                # Cannot decode to anything reasonable
                word_converted_punct.append(c)
            else:
                # Check if all punctuation and therefore fine
                # to include unidecoded version
                allowed_punct = punct_word(
                    decoded_c,
                    punctuation=ALLOWED_CONVERTED_UNICODE_PUNCTUATION)

                if allowed_punct:
                    word_converted_punct.append(decoded_c)
                else:
                    word_converted_punct.append(c)
        return ''.join(word_converted_punct)

    def convert_unicode_word(self, word):
        """ Converts Unicode words to ASCII using unidecode. If Unicode is not
            allowed (set as a variable during initialization), then only
            punctuation that can be converted to ASCII will be allowed.
        """
        if self.check_ascii(word):
            return True, word

        # First we ensure that the Unicode is normalized so it's
        # always a single character.
        word = unicodedata.normalize("NFKC", word)

        # Convert Unicode punctuation to ASCII equivalent. We want
        # e.g. u"\u203c" (double exclamation mark) to be treated the same
        # as u"!!" no matter if we allow other Unicode characters or not.
        word = self.convert_unicode_punctuation(word)

        if self.ignore_emojis:
            _, word = separate_emojis_and_text(word)

        # If conversion of punctuation and removal of emojis took care
        # of all the Unicode or if we allow Unicode then everything is fine
        if self.check_ascii(word) or self.allow_unicode_text:
            return True, word
        else:
            # Sometimes we might want to simply ignore Unicode sentences
            # (e.g. for vocabulary creation). This is another way to prevent
            # "polution" of strange Unicode tokens from low quality datasets
            return False, ''

    def data_preprocess_filtering(self, line, iter_i):
        """ To be overridden with specific preprocessing/filtering behavior
            if desired.

            Returns a boolean of whether the line should be accepted and the
            preprocessed text.

            Runs prior to tokenization.
        """
        return True, line, {}

    def data_postprocess_filtering(self, words, iter_i):
        """ To be overridden with specific postprocessing/filtering behavior
            if desired.

            Returns a boolean of whether the line should be accepted and the
            postprocessed text.

            Runs after tokenization.
        """
        return True, words, {}

    def extract_valid_sentence_words(self, line):
        """ Line may either a string of a list of strings depending on how
            the stream is being parsed.
            Domain-specific processing and filtering can be done both prior to
            and after tokenization.
            Custom information about the line can be extracted during the
            processing phases and returned as a dict.
        """

        info = {}

        pre_valid, pre_line, pre_info = \
            self.data_preprocess_filtering(line, self.stats['total'])
        info.update(pre_info)
        if not pre_valid:
            self.stats['pretokenization_filtered'] += 1
            return False, [], info

        words = self.get_words(pre_line)
        if len(words) == 0:
            self.stats['unicode_filtered'] += 1
            return False, [], info

        post_valid, post_words, post_info = \
            self.data_postprocess_filtering(words, self.stats['total'])
        info.update(post_info)
        if not post_valid:
            self.stats['posttokenization_filtered'] += 1
        return post_valid, post_words, info

    def generate_array_from_input(self):
        sentences = []
        for words in self:
            sentences.append(words)
        return sentences

    def reset_stats(self):
        self.stats = {'pretokenization_filtered': 0,
                      'unicode_filtered': 0,
                      'posttokenization_filtered': 0,
                      'total': 0,
                      'valid': 0}

    def __iter__(self):
        if self.stream is None:
            raise ValueError("Stream should be set before iterating over it!")

        for line in self.stream:
            valid, words, info = self.extract_valid_sentence_words(line)

            # Words may be filtered away due to unidecode etc.
            # In that case the words should not be passed on.
            if valid and len(words):
                self.stats['valid'] += 1
                yield words, info

            self.stats['total'] += 1


class TweetWordGenerator(WordGenerator):
    ''' Returns np array or generator of ASCII sentences for given tweet input.
        Any file opening/closing should be handled outside of this class.
    '''

    def __init__(self, stream, wanted_emojis=None, english_words=None,
                 non_english_user_set=None, allow_unicode_text=False,
                 ignore_retweets=True, ignore_url_tweets=True,
                 ignore_mention_tweets=False):

        self.wanted_emojis = wanted_emojis
        self.english_words = english_words
        self.non_english_user_set = non_english_user_set
        self.ignore_retweets = ignore_retweets
        self.ignore_url_tweets = ignore_url_tweets
        self.ignore_mention_tweets = ignore_mention_tweets
        WordGenerator.__init__(self, stream,
                               allow_unicode_text=allow_unicode_text)

    def validated_tweet(self, data):
        ''' A bunch of checks to determine whether the tweet is valid.
            Also returns emojis contained by the tweet.
        '''

        # Ordering of validations is important for speed
        # If it passes all checks, then the tweet is validated for usage

        # Skips incomplete tweets
        if len(data) <= 9:
            return False, []

        text = data[9]

        if self.ignore_retweets and RETWEETS_RE.search(text):
            return False, []

        if self.ignore_url_tweets and URLS_RE.search(text):
            return False, []

        if self.ignore_mention_tweets and MENTION_RE.search(text):
            return False, []

        if self.wanted_emojis is not None:
            uniq_emojis = np.unique(extract_emojis(text, self.wanted_emojis))
            if len(uniq_emojis) == 0:
                return False, []
        else:
            uniq_emojis = []

        if self.non_english_user_set is not None and \
           non_english_user(data[1], self.non_english_user_set):
            return False, []
        return True, uniq_emojis

    def data_preprocess_filtering(self, line, iter_i):
        fields = line.strip().split("\t")
        valid, emojis = self.validated_tweet(fields)
        text = fields[9].replace(u'\\n', u'') \
                        .replace(u'\\r', u'') \
                        .replace(u'&amp', u'&') if valid else ''
        return valid, text, {'emojis': emojis}

    def data_postprocess_filtering(self, words, iter_i):
        valid_length = correct_length(words, 1, None)
        valid_english, n_words, n_english = mostly_english(words,
                                                           self.english_words)
        if valid_length and valid_english:
            return True, words, {'length': len(words),
                                 'n_normal_words': n_words,
                                 'n_english': n_english}
        else:
            return False, [], {'length': len(words),
                               'n_normal_words': n_words,
                               'n_english': n_english}
