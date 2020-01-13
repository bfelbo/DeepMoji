import json
from os.path import dirname, abspath, join

import numpy as np
from nose.plugins.attrib import attr

from deepmoji.class_avg_finetuning import relabel
from deepmoji.finetuning import (
    calculate_batchsize_maxlen,
    freeze_layers,
    change_trainable,
    relabel,
    finetune,
    load_benchmark
)
from deepmoji.global_variables import (
    PRETRAINED_PATH,
    NB_TOKENS,
    get_vocabulary,
)
from deepmoji.model_def import (
    deepmoji_transfer,
    deepmoji_feature_encoding,
    deepmoji_emojis
)
from deepmoji.sentence_tokenizer import SentenceTokenizer


def test_calculate_batchsize_maxlen():
    """ Batch size and max length are calculated properly.
    """
    texts = ['a b c d', 'e f g h i']
    batch_size, maxlen = calculate_batchsize_maxlen(texts)
    assert batch_size == 250
    assert maxlen == 10


def test_freeze_layers():
    """ Correct layers are frozen.
    """
    model = deepmoji_transfer(5, 30)
    keyword = 'softmax'

    model = freeze_layers(model, unfrozen_keyword=keyword)

    for layer in model.layers:
        if layer is not None and len(layer.trainable_weights):
            if keyword in layer.name:
                assert layer.trainable
            else:
                assert not layer.trainable


def test_change_trainable():
    """ change_trainable() changes trainability of layers.
    """
    model = deepmoji_transfer(5, 30)
    change_trainable(model.layers[0], False)
    assert not model.layers[0].trainable
    change_trainable(model.layers[0], True)
    assert model.layers[0].trainable


def test_deepmoji_transfer_extend_embedding():
    """ Defining deepmoji with extension.
    """
    extend_with = 50
    model = deepmoji_transfer(5, 30, weight_path=PRETRAINED_PATH,
                              extend_embedding=extend_with)
    embedding_layer = model.layers[1]
    assert embedding_layer.input_dim == NB_TOKENS + extend_with


def test_deepmoji_return_attention():
    # test the output of the normal model
    model = deepmoji_emojis(maxlen=30, weight_path=PRETRAINED_PATH)
    # check correct number of outputs
    assert 1 == len(model.outputs)
    # check model outputs come from correct layers
    assert [['softmax', 0, 0]] == model.get_config()['output_layers']
    # ensure that output shapes are correct (assume a 5-example batch of 30-timesteps)
    input_shape = (5, 30, 2304)
    assert (5, 2304) == model.layers[6].compute_output_shape(input_shape)

    # repeat above described tests when returning attention weights
    model = deepmoji_emojis(maxlen=30, weight_path=PRETRAINED_PATH, return_attention=True)
    assert 2 == len(model.outputs)
    assert [['softmax', 0, 0], ['attlayer', 0, 1]] == model.get_config()['output_layers']
    assert [(5, 2304), (5, 30)] == model.layers[6].compute_output_shape(input_shape)


def test_relabel():
    """ relabel() works with multi-class labels.
    """
    nb_classes = 3
    inputs = np.array([
        [True, False, False],
        [False, True, False],
        [True, False, True],
    ])
    expected_0 = np.array([True, False, True])
    expected_1 = np.array([False, True, False])
    expected_2 = np.array([False, False, True])

    assert np.array_equal(relabel(inputs, 0, nb_classes), expected_0)
    assert np.array_equal(relabel(inputs, 1, nb_classes), expected_1)
    assert np.array_equal(relabel(inputs, 2, nb_classes), expected_2)


def test_relabel_binary():
    """ relabel() works with binary classification (no changes to labels)
    """
    nb_classes = 2
    inputs = np.array([True, False, False])

    assert np.array_equal(relabel(inputs, 0, nb_classes), inputs)


DATA_DIR = join(dirname(dirname(abspath(__file__))), 'data')
SS_YOUTUBE_DATASET_DIR = join(DATA_DIR, "SS-Youtube")
SS_YOUTUBE_DATASET_PATH = join(SS_YOUTUBE_DATASET_DIR, "raw.pickle")


@attr('slow')
def test_finetune_full():
    """ finetuning using 'full'.
    """
    nb_classes = 2
    min_acc = 0.65

    data = load_benchmark(SS_YOUTUBE_DATASET_PATH, get_vocabulary(), extend_with=10000)
    model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH,
                              extend_embedding=data['added'])
    model.summary()
    model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                          data['batch_size'], method='full', nb_epochs=1)

    print("Finetune full SS-Youtube 1 epoch acc: {}".format(acc))
    assert acc >= min_acc


@attr('slow')
def test_finetune_last():
    """ finetuning using 'last'.
    """
    nb_classes = 2
    min_acc = 0.65

    data = load_benchmark(SS_YOUTUBE_DATASET_PATH, get_vocabulary())

    model = deepmoji_transfer(nb_classes, data['maxlen'], PRETRAINED_PATH)
    model.summary()
    model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                          data['batch_size'], method='last', nb_epochs=1)

    print(("Finetune last SS-Youtube 1 epoch acc: {}".format(acc)))
    assert acc >= min_acc


TEST_SENTENCES = [
    'I love mom\'s cooking',
    'I love how you never reply back..',
    'I love cruising with my homies',
    'I love messing with yo mind!!',
    'I love you and now you\'re just gone..',
    'This is shit',
    'This is the shit'
]


def top_elements(array, k):
    """Given an array of emoji probabilities get the top k emoji indexes
    for the highest probability emojis"""
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


def test_score_emoji():
    """ Emoji predictions make sense.
    """
    maxlen = 30
    expected = [
        np.array([36, 4, 8, 16, 47]),
        np.array([1, 19, 55, 25, 46]),
        np.array([31, 6, 30, 15, 13]),
        np.array([54, 44, 9, 50, 49]),
        np.array([46, 5, 27, 35, 34]),
        np.array([55, 32, 27, 1, 37]),
        np.array([48, 11, 6, 31, 9])
    ]

    st = SentenceTokenizer(get_vocabulary(), fixed_length=maxlen)
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    model = deepmoji_emojis(maxlen=maxlen, weight_path=PRETRAINED_PATH)
    prob = model.predict(tokenized)
    # Find top emojis for each sentence
    for i, t_prob in enumerate(prob):
        assert np.array_equal(top_elements(t_prob, 5), expected[i])


def test_encode_texts():
    """ Text encoding is stable.
    """
    maxlen = 30
    st = SentenceTokenizer(get_vocabulary(), fixed_length=maxlen)
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    model = deepmoji_feature_encoding(maxlen=maxlen, weight_path=PRETRAINED_PATH)
    prob = model.predict(tokenized)
    avg_across_sentences = np.around(np.mean(prob, axis=0)[:5], 3)
    assert np.allclose(avg_across_sentences, np.array([-0.023, 0.021, -0.037, -0.001, -0.005]))
