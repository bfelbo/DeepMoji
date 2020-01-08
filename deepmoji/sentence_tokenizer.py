'''
Provides functionality for converting a given list of tokens (words) into
numbers, according to the given vocabulary.
'''


import numbers
import numpy as np
from .create_vocab import extend_vocab, VocabBuilder
from .word_generator import WordGenerator
from .global_variables import SPECIAL_TOKENS
from sklearn.model_selection import train_test_split
from copy import deepcopy


class SentenceTokenizer():
    """ Create numpy array of tokens corresponding to input sentences.
        The vocabulary can include Unicode tokens.
    """

    def __init__(self, vocabulary, fixed_length, custom_wordgen=None,
                 ignore_sentences_with_only_custom=False, masking_value=0,
                 unknown_value=1):
        """ Needs a dictionary as input for the vocabulary.
        """

        if len(vocabulary) > np.iinfo('uint16').max:
            raise ValueError('Dictionary is too big ({} tokens) for the numpy '
                             'datatypes used (max limit={}). Reduce vocabulary'
                             ' or adjust code accordingly!'
                             .format(len(vocabulary), np.iinfo('uint16').max))

        # Shouldn't be able to modify the given vocabulary
        self.vocabulary = deepcopy(vocabulary)
        self.fixed_length = fixed_length
        self.ignore_sentences_with_only_custom = ignore_sentences_with_only_custom
        self.masking_value = masking_value
        self.unknown_value = unknown_value

        # Initialized with an empty stream of sentences that must then be fed
        # to the generator at a later point for reusability.
        # A custom word generator can be used for domain-specific filtering etc
        if custom_wordgen is not None:
            assert custom_wordgen.stream is None
            self.wordgen = custom_wordgen
            self.uses_custom_wordgen = True
        else:
            self.wordgen = WordGenerator(None, allow_unicode_text=True,
                                         ignore_emojis=False,
                                         remove_variation_selectors=True,
                                         break_replacement=True)
            self.uses_custom_wordgen = False

    def tokenize_sentences(self, sentences, reset_stats=True, max_sentences=None):
        """ Converts a given list of sentences into a numpy array according to
            its vocabulary.

        # Arguments:
            sentences: List of sentences to be tokenized.
            reset_stats: Whether the word generator's stats should be reset.
            max_sentences: Maximum length of sentences. Must be set if the
                length cannot be inferred from the input.

        # Returns:
            Numpy array of the tokenization sentences with masking,
            infos,
            stats

        # Raises:
            ValueError: When maximum length is not set and cannot be inferred.
        """

        if max_sentences is None and not hasattr(sentences, '__len__'):
            raise ValueError('Either you must provide an array with a length'
                             'attribute (e.g. a list) or specify the maximum '
                             'length yourself using `max_sentences`!')
        n_sentences = (max_sentences if max_sentences is not None
                       else len(sentences))

        if self.masking_value == 0:
            tokens = np.zeros((n_sentences, self.fixed_length), dtype='uint16')
        else:
            tokens = (np.ones((n_sentences, self.fixed_length), dtype='uint16') *
                      self.masking_value)

        if reset_stats:
            self.wordgen.reset_stats()

        # With a custom word generator info can be extracted from each
        # sentence (e.g. labels)
        infos = []

        # Returns words as strings and then map them to vocabulary
        self.wordgen.stream = sentences
        next_insert = 0
        n_ignored_unknowns = 0
        for s_words, s_info in self.wordgen:
            s_tokens = self.find_tokens(s_words)

            if (self.ignore_sentences_with_only_custom and
                np.all([True if t < len(SPECIAL_TOKENS)
                        else False for t in s_tokens])):
                n_ignored_unknowns += 1
                continue
            if len(s_tokens) > self.fixed_length:
                s_tokens = s_tokens[:self.fixed_length]
            tokens[next_insert, :len(s_tokens)] = s_tokens
            infos.append(s_info)
            next_insert += 1

        # For standard word generators all sentences should be tokenized
        # this is not necessarily the case for custom wordgenerators as they
        # may filter the sentences etc.
        if not self.uses_custom_wordgen and not self.ignore_sentences_with_only_custom:
            assert len(sentences) == next_insert
        else:
            # adjust based on actual tokens received
            tokens = tokens[:next_insert]
            infos = infos[:next_insert]
        return tokens, infos, self.wordgen.stats

    def find_tokens(self, words):
        assert len(words) > 0
        tokens = []
        for w in words:
            try:
                tokens.append(self.vocabulary[w])
            except KeyError:
                tokens.append(self.unknown_value)
        return tokens

    def split_train_val_test(self, sentences, info_dicts,
                             split_parameter=[0.7, 0.1, 0.2], extend_with=0):
        """ Splits given sentences into three different datasets: training,
            validation and testing.

        # Arguments:
            sentences: The sentences to be tokenized.
            info_dicts: A list of dicts that contain information about each
                sentence (e.g. a label).
            split_parameter: A parameter for deciding the splits between the
                three different datasets. If instead of being passed three
                values, three lists are passed, then these will be used to
                specify which observation belong to which dataset.
            extend_with: An optional parameter. If > 0 then this is the number
                of tokens added to the vocabulary from this dataset. The
                expanded vocab will be generated using only the training set,
                but is applied to all three sets.

        # Returns:
            List of three lists of tokenized sentences,

            List of three corresponding dictionaries with information,

            How many tokens have been added to the vocab. Make sure to extend
            the embedding layer of the model accordingly.
        """

        # If passed three lists, use those directly
        if isinstance(split_parameter, list) and \
                all(isinstance(x, list) for x in split_parameter) and \
                len(split_parameter) == 3:

            # Helper function to verify provided indices are numbers in range
            def verify_indices(inds):
                return list([i for i in inds if isinstance(i, numbers.Number) and
                                   i < len(sentences)])

            ind_train = verify_indices(split_parameter[0])
            ind_val = verify_indices(split_parameter[1])
            ind_test = verify_indices(split_parameter[2])
        else:
            # Split sentences and dicts
            ind = list(range(len(sentences)))
            ind_train, ind_test = train_test_split(ind, test_size=split_parameter[2])
            ind_train, ind_val = train_test_split(ind_train, test_size=split_parameter[1])

        # Map indices to data
        train = np.array([sentences[x] for x in ind_train])
        test = np.array([sentences[x] for x in ind_test])
        val = np.array([sentences[x] for x in ind_val])

        info_train = np.array([info_dicts[x] for x in ind_train])
        info_test = np.array([info_dicts[x] for x in ind_test])
        info_val = np.array([info_dicts[x] for x in ind_val])

        added = 0
        # Extend vocabulary with training set tokens
        if extend_with > 0:
            wg = WordGenerator(train)
            vb = VocabBuilder(wg)
            vb.count_all_words()
            added = extend_vocab(self.vocabulary, vb, max_tokens=extend_with)

        # Wrap results
        result = [self.tokenize_sentences(s)[0] for s in [train, val, test]]
        result_infos = [info_train, info_val, info_test]

        return result, result_infos, added

    def to_sentence(self, sentence_idx):
        """ Converts a tokenized sentence back to a list of words.

        # Arguments:
            sentence_idx: List of numbers, representing a tokenized sentence
                given the current vocabulary.

        # Returns:
            String created by converting all numbers back to words and joined
            together with spaces.
        """
        # Have to recalculate the mappings in case the vocab was extended.
        ind_to_word = {ind: word for word, ind in self.vocabulary.items()}

        sentence_as_list = [ind_to_word[x] for x in sentence_idx]
        cleaned_list = [x for x in sentence_as_list if x != 'CUSTOM_MASK']
        return " ".join(cleaned_list)


def coverage(dataset, verbose=False):
    """ Computes the percentage of words in a given dataset that are unknown.

    # Arguments:
        dataset: Tokenized dataset to be checked.
        verbose: Verbosity flag.

    # Returns:
        Percentage of unknown tokens.
    """
    n_total = np.count_nonzero(dataset)
    n_unknown = np.sum(dataset == 1)
    coverage = 1.0 - float(n_unknown) / n_total

    if verbose:
        print("Unknown words: {}".format(n_unknown))
        print("Total words: {}".format(n_total))
        print("Coverage: {}".format(coverage))
    return coverage
