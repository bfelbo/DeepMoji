""" Model definition functions and weight loading.
"""

from __future__ import print_function, division

from keras.models import Model, Sequential
from keras.layers.merge import concatenate
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation
from keras.regularizers import L1L2
from attlayer import AttentionWeightedAverage
from global_variables import NB_TOKENS, NB_EMOJI_CLASSES
import numpy as np
from copy import deepcopy
from os.path import exists
import h5py


def deepmoji_feature_encoding(maxlen, weight_path, return_attention=False):
    """ Loads the pretrained DeepMoji model for extracting features
        from the penultimate feature layer. In this way, it transforms
        the text into its emotional encoding.

    # Arguments:
        maxlen: Maximum length of a sentence (given in tokens).
        weight_path: Path to model weights to be loaded.
        return_attention: If true, output will be weight of each input token
            used for the prediction

    # Returns:
        Pretrained model for encoding text into feature vectors.
    """

    model = deepmoji_architecture(nb_classes=None, nb_tokens=NB_TOKENS,
                                  maxlen=maxlen, feature_output=True,
                                  return_attention=return_attention)
    load_specific_weights(model, weight_path, exclude_names=['softmax'])
    return model


def deepmoji_emojis(maxlen, weight_path, return_attention=False):
    """ Loads the pretrained DeepMoji model for extracting features
        from the penultimate feature layer. In this way, it transforms
        the text into its emotional encoding.

    # Arguments:
        maxlen: Maximum length of a sentence (given in tokens).
        weight_path: Path to model weights to be loaded.
        return_attention: If true, output will be weight of each input token
            used for the prediction

    # Returns:
        Pretrained model for encoding text into feature vectors.
    """

    model = deepmoji_architecture(nb_classes=NB_EMOJI_CLASSES,
                                  nb_tokens=NB_TOKENS, maxlen=maxlen,
                                  return_attention=return_attention)
    model.load_weights(weight_path, by_name=False)
    return model


def deepmoji_transfer(nb_classes, maxlen, weight_path=None, extend_embedding=0,
                      embed_dropout_rate=0.25, final_dropout_rate=0.5,
                      embed_l2=1E-6):
    """ Loads the pretrained DeepMoji model for finetuning/transfer learning.
        Does not load weights for the softmax layer.

        Note that if you are planning to use class average F1 for evaluation,
        nb_classes should be set to 2 instead of the actual number of classes
        in the dataset, since binary classification will be performed on each
        class individually.

        Note that for the 'new' method, weight_path should be left as None.

    # Arguments:
        nb_classes: Number of classes in the dataset.
        maxlen: Maximum length of a sentence (given in tokens).
        weight_path: Path to model weights to be loaded.
        extend_embedding: Number of tokens that have been added to the
            vocabulary on top of NB_TOKENS. If this number is larger than 0,
            the embedding layer's dimensions are adjusted accordingly, with the
            additional weights being set to random values.
        embed_dropout_rate: Dropout rate for the embedding layer.
        final_dropout_rate: Dropout rate for the final Softmax layer.
        embed_l2: L2 regularization for the embedding layerl.

    # Returns:
        Model with the given parameters.
    """

    model = deepmoji_architecture(nb_classes=nb_classes,
                                  nb_tokens=NB_TOKENS + extend_embedding,
                                  maxlen=maxlen, embed_dropout_rate=embed_dropout_rate,
                                  final_dropout_rate=final_dropout_rate, embed_l2=embed_l2)

    if weight_path is not None:
        load_specific_weights(model, weight_path,
                              exclude_names=['softmax'],
                              extend_embedding=extend_embedding)
    return model


def deepmoji_architecture(nb_classes, nb_tokens, maxlen, feature_output=False, embed_dropout_rate=0, final_dropout_rate=0, embed_l2=1E-6, return_attention=False):
    """
    Returns the DeepMoji architecture uninitialized and
    without using the pretrained model weights.

    # Arguments:
        nb_classes: Number of classes in the dataset.
        nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
        maxlen: Maximum length of a token.
        feature_output: If True the model returns the penultimate
                        feature vector rather than Softmax probabilities
                        (defaults to False).
        embed_dropout_rate: Dropout rate for the embedding layer.
        final_dropout_rate: Dropout rate for the final Softmax layer.
        embed_l2: L2 regularization for the embedding layerl.

    # Returns:
        Model with the given parameters.
    """
    # define embedding layer that turns word tokens into vectors
    # an activation function is used to bound the values of the embedding
    model_input = Input(shape=(maxlen,), dtype='int32')
    embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
    embed = Embedding(input_dim=nb_tokens,
                      output_dim=256,
                      mask_zero=True,
                      input_length=maxlen,
                      embeddings_regularizer=embed_reg,
                      name='embedding')
    x = embed(model_input)
    x = Activation('tanh')(x)

    # entire embedding channels are dropped out instead of the
    # normal Keras embedding dropout, which drops all channels for entire words
    # many of the datasets contain so few words that losing one or more words can alter the emotions completely
    if embed_dropout_rate != 0:
        embed_drop = SpatialDropout1D(embed_dropout_rate, name='embed_drop')
        x = embed_drop(x)

    # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
    # ordering of the way the merge is done is important for consistency with the pretrained model
    lstm_0_output = Bidirectional(LSTM(512, return_sequences=True), name="bi_lstm_0")(x)
    lstm_1_output = Bidirectional(LSTM(512, return_sequences=True), name="bi_lstm_1")(lstm_0_output)
    x = concatenate([lstm_1_output, lstm_0_output, x])

    # if return_attention is True in AttentionWeightedAverage, an additional tensor
    # representing the weight at each timestep is returned
    weights = None
    x = AttentionWeightedAverage(name='attlayer', return_attention=return_attention)(x)
    if return_attention:
        x, weights = x

    if not feature_output:
        # output class probabilities
        if final_dropout_rate != 0:
            x = Dropout(final_dropout_rate)(x)

        if nb_classes > 2:
            outputs = [Dense(nb_classes, activation='softmax', name='softmax')(x)]
        else:
            outputs = [Dense(1, activation='sigmoid', name='softmax')(x)]
    else:
        # output penultimate feature vector
        outputs = [x]

    if return_attention:
        # add the attention weights to the outputs if required
        outputs.append(weights)

    return Model(inputs=[model_input], outputs=outputs, name="DeepMoji")


def load_specific_weights(model, weight_path, exclude_names=[], extend_embedding=0, verbose=True):
    """ Loads model weights from the given file path, excluding any
        given layers.

    # Arguments:
        model: Model whose weights should be loaded.
        weight_path: Path to file containing model weights.
        exclude_names: List of layer names whose weights should not be loaded.
        extend_embedding: Number of new words being added to vocabulary.
        verbose: Verbosity flag.

    # Raises:
        ValueError if the file at weight_path does not exist.
    """
    if not exists(weight_path):
        raise ValueError('ERROR (load_weights): The weights file at {} does '
                         'not exist. Refer to the README for instructions.'
                         .format(weight_path))

    if extend_embedding and 'embedding' in exclude_names:
        raise ValueError('ERROR (load_weights): Cannot extend a vocabulary '
                         'without loading the embedding weights.')

    # Copy only weights from the temporary model that are wanted
    # for the specific task (e.g. the Softmax is often ignored)
    layer_weights = get_weights_from_hdf5(weight_path)
    for i, w in enumerate(layer_weights):
        l_name = w[0]
        weight_names = w[1]
        weight_values = w[2]

        if l_name in exclude_names:
            if verbose:
                print('Ignoring weights for {}'.format(l_name))
            continue

        try:
            model_l = model.get_layer(name=l_name)
        except ValueError:
            raise ValueError("Weights had layer {},".format(l_name) +
                             " but could not find this layer in model.")

        if verbose:
            print('Loading weights for {}'.format(l_name))

        # extend embedding layer to allow new randomly initialized words
        # if requested. Otherwise, just load the weights for the layer.
        if type(model_l) is Embedding and extend_embedding > 0:
            comb_weights = append_to_embedding(weight_values,
                                               model_l.get_weights())
            model_l.set_weights(comb_weights)
            if verbose:
                print('Extended vocabulary for embedding layer ' +
                      'from {} to {} tokens.'.format(
                          NB_TOKENS, NB_TOKENS + extend_embedding))
        else:
            model_l.set_weights(weight_values)


def append_to_embedding(pretrain_weights, random_init_weights):
    """ Uses pretrained weights for the tokens already in the vocabulary.
        Remaining weights will be left with the random initialization. """

    pretrain_weights = deepcopy(pretrain_weights)
    if type(pretrain_weights) == list:
        pretrain_weights = pretrain_weights[0]
    if type(random_init_weights) == list:
        random_init_weights = random_init_weights[0]

    nb_old_tokens = np.shape(pretrain_weights)[0]
    random_init_weights[:nb_old_tokens] = pretrain_weights

    # must be returned as a list to be properly inserted into Keras model
    return [random_init_weights]


def get_weights_from_hdf5(filepath):
    """ Loads the weights from a saved Keras model into numpy arrays.
        The weights are saved using Keras 2.0 so we don't need all the
        conversion functionality for handling old weights.
    """

    with h5py.File(filepath, mode='r') as f:
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        layer_weights = []
        for k, l_name in enumerate(layer_names):
            g = f[l_name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name][:] for weight_name in weight_names]
            if len(weight_values):
                layer_weights.append([l_name, weight_names, weight_values])
        return layer_weights
