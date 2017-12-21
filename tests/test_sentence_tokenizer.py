from __future__ import print_function
import test_helper
import json

from deepmoji.sentence_tokenizer import SentenceTokenizer

sentences = [u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J']

dicts = [
    {'label': 0},
    {'label': 1},
    {'label': 2},
    {'label': 3},
    {'label': 4},
    {'label': 5},
    {'label': 6},
    {'label': 7},
    {'label': 8},
    {'label': 9},
]

train_ind = [0, 5, 3, 6, 8]
val_ind = [9, 2, 1]
test_ind = [4, 7]

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)


def test_dataset_split_parameter():
    """ Dataset is split in the desired ratios
    """
    split_parameter = [0.7, 0.1, 0.2]
    st = SentenceTokenizer(vocab, 30)

    result, result_dicts, _ = st.split_train_val_test(sentences, dicts,
                                                      split_parameter, extend_with=0)
    train = result[0]
    val = result[1]
    test = result[2]

    train_dicts = result_dicts[0]
    val_dicts = result_dicts[1]
    test_dicts = result_dicts[2]

    assert len(train) == len(sentences) * split_parameter[0]
    assert len(val) == len(sentences) * split_parameter[1]
    assert len(test) == len(sentences) * split_parameter[2]

    assert len(train_dicts) == len(dicts) * split_parameter[0]
    assert len(val_dicts) == len(dicts) * split_parameter[1]
    assert len(test_dicts) == len(dicts) * split_parameter[2]


def test_dataset_split_explicit():
    """ Dataset is split according to given indices
    """
    split_parameter = [train_ind, val_ind, test_ind]
    st = SentenceTokenizer(vocab, 30)
    tokenized, _, _ = st.tokenize_sentences(sentences)

    result, result_dicts, added = st.split_train_val_test(sentences, dicts, split_parameter, extend_with=0)
    train = result[0]
    val = result[1]
    test = result[2]

    train_dicts = result_dicts[0]
    val_dicts = result_dicts[1]
    test_dicts = result_dicts[2]

    for i, sentence in enumerate(sentences):
        if i in train_ind:
            assert tokenized[i] in train
            assert dicts[i] in train_dicts
        elif i in val_ind:
            assert tokenized[i] in val
            assert dicts[i] in val_dicts
        elif i in test_ind:
            assert tokenized[i] in test
            assert dicts[i] in test_dicts

    assert len(train) == len(train_ind)
    assert len(val) == len(val_ind)
    assert len(test) == len(test_ind)
    assert len(train_dicts) == len(train_ind)
    assert len(val_dicts) == len(val_ind)
    assert len(test_dicts) == len(test_ind)


def test_id_to_sentence():
    """Tokenizing and converting back preserves the input.
    """
    vb = {'CUSTOM_MASK': 0,
          'aasdf': 1000,
          'basdf': 2000}

    sentence = u'aasdf basdf basdf basdf'
    st = SentenceTokenizer(vb, 30)
    token, _, _ = st.tokenize_sentences([sentence])
    assert st.to_sentence(token[0]) == sentence


def test_id_to_sentence_with_unknown():
    """Tokenizing and converting back preserves the input, except for unknowns.
    """
    vb = {'CUSTOM_MASK': 0,
          'CUSTOM_UNKNOWN': 1,
          'aasdf': 1000,
          'basdf': 2000}

    sentence = u'aasdf basdf ccc'
    expected = u'aasdf basdf CUSTOM_UNKNOWN'
    st = SentenceTokenizer(vb, 30)
    token, _, _ = st.tokenize_sentences([sentence])
    assert st.to_sentence(token[0]) == expected
