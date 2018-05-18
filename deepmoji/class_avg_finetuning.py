""" Class average finetuning functions. Before using any of these finetuning
    functions, ensure that the model is set up with nb_classes=2.
"""
from __future__ import print_function

import sys
import uuid
import numpy as np
from os.path import dirname
from time import sleep
from keras.optimizers import Adam

from global_variables import (
    FINETUNING_METHODS,
    WEIGHTS_DIR)
from finetuning import (
    freeze_layers,
    sampling_generator,
    finetuning_callbacks,
    train_by_chain_thaw,
    find_f1_threshold)


def relabel(y, current_label_nr, nb_classes):
    """ Makes a binary classification for a specific class in a
        multi-class dataset.

    # Arguments:
        y: Outputs to be relabelled.
        current_label_nr: Current label number.
        nb_classes: Total number of classes.

    # Returns:
        Relabelled outputs of a given multi-class dataset into a binary
        classification dataset.
    """

    # Handling binary classification
    if nb_classes == 2 and len(y.shape) == 1:
        return y

    y_new = np.zeros(len(y))
    y_cut = y[:, current_label_nr]
    label_pos = np.where(y_cut == 1)[0]
    y_new[label_pos] = 1
    return y_new


def class_avg_finetune(model, texts, labels, nb_classes, batch_size,
                       method, epoch_size=5000,
                       nb_epochs=1000, error_checking=True,
                       verbose=True):
    """ Compiles and finetunes the given model.

    # Arguments:
        model: Model to be finetuned
        texts: List of three lists, containing tokenized inputs for training,
            validation and testing (in that order).
        labels: List of three lists, containing labels for training,
            validation and testing (in that order).
        nb_classes: Number of classes in the dataset.
        batch_size: Batch size.
        method: Finetuning method to be used. For available methods, see
            FINETUNING_METHODS in global_variables.py. Note that the model
            should be defined accordingly (see docstring for deepmoji_transfer())
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs. Doesn't matter much as early stopping is used.
        error_checking: If set to True, warnings will be printed when the label
            list has the wrong dimensions.
        verbose: Verbosity flag.

    # Returns:
        Model after finetuning,
        score after finetuning using the class average F1 metric.
    """

    if method not in FINETUNING_METHODS:
        raise ValueError('ERROR (class_avg_tune_trainable): '
                         'Invalid method parameter. '
                         'Available options: {}'.format(FINETUNING_METHODS))

    (X_train, y_train) = (texts[0], labels[0])
    (X_val, y_val) = (texts[1], labels[1])
    (X_test, y_test) = (texts[2], labels[2])

    checkpoint_path = '{}/deepmoji-checkpoint-{}.hdf5' \
                      .format(WEIGHTS_DIR, str(uuid.uuid4()))

    f1_init_path = '{}/deepmoji-f1-init-{}.hdf5' \
                   .format(WEIGHTS_DIR, str(uuid.uuid4()))

    # Check dimension of labels
    if error_checking:
        # Binary classification has two classes but one value
        expected_shape = 1 if nb_classes == 2 else nb_classes

        for ls in [y_train, y_val, y_test]:
            if len(ls.shape) <= 1 or not ls.shape[1] == expected_shape:
                print('WARNING (class_avg_tune_trainable): '
                      'The dimension of the provided '
                      'labels do not match the expected value. '
                      'Expected: {}, actual: {}'
                      .format(expected_shape, ls.shape[1]))
                break

    if method in ['last', 'new']:
        lr = 0.001
    elif method in ['full', 'chain-thaw']:
        lr = 0.0001

    loss = 'binary_crossentropy'

    # Freeze layers if using last
    if method == 'last':
        model = freeze_layers(model, unfrozen_keyword='softmax')

    # Compile model, for chain-thaw we compile it later (after freezing)
    if method != 'chain-thaw':
        adam = Adam(clipnorm=1, lr=lr)
        model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])

    # Training
    if verbose:
        print('Method:  {}'.format(method))
        print('Classes: {}'.format(nb_classes))

    if method == 'chain-thaw':
        result = class_avg_chainthaw(model, nb_classes=nb_classes,
                                     train=(X_train, y_train),
                                     val=(X_val, y_val),
                                     test=(X_test, y_test),
                                     batch_size=batch_size, loss=loss,
                                     epoch_size=epoch_size,
                                     nb_epochs=nb_epochs,
                                     checkpoint_weight_path=checkpoint_path,
                                     f1_init_weight_path=f1_init_path,
                                     verbose=verbose)
    else:
        result = class_avg_tune_trainable(model, nb_classes=nb_classes,
                                          train=(X_train, y_train),
                                          val=(X_val, y_val),
                                          test=(X_test, y_test),
                                          epoch_size=epoch_size,
                                          nb_epochs=nb_epochs,
                                          batch_size=batch_size,
                                          init_weight_path=f1_init_path,
                                          checkpoint_weight_path=checkpoint_path,
                                          verbose=verbose)
    return model, result


def prepare_labels(y_train, y_val, y_test, iter_i, nb_classes):
    # Relabel into binary classification
    y_train_new = relabel(y_train, iter_i, nb_classes)
    y_val_new = relabel(y_val, iter_i, nb_classes)
    y_test_new = relabel(y_test, iter_i, nb_classes)
    return y_train_new, y_val_new, y_test_new


def prepare_generators(X_train, y_train_new, X_val, y_val_new, batch_size, epoch_size):
    # Create sample generators
    # Make a fixed validation set to avoid fluctuations in validation
    train_gen = sampling_generator(X_train, y_train_new, batch_size,
                                   upsample=False)
    val_gen = sampling_generator(X_val, y_val_new,
                                 epoch_size, upsample=False)
    X_val_resamp, y_val_resamp = next(val_gen)
    return train_gen, X_val_resamp, y_val_resamp


def class_avg_tune_trainable(model, nb_classes, train, val, test, epoch_size,
                             nb_epochs, batch_size, init_weight_path,
                             checkpoint_weight_path, patience=5,
                             verbose=True):
    """ Finetunes the given model using the F1 measure.

    # Arguments:
        model: Model to be finetuned.
        nb_classes: Number of classes in the given dataset.
        train: Training data, given as a tuple of (inputs, outputs)
        val: Validation data, given as a tuple of (inputs, outputs)
        test: Testing data, given as a tuple of (inputs, outputs)
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        batch_size: Batch size.
        init_weight_path: Filepath where weights will be initially saved before
            training each class. This file will be rewritten by the function.
        checkpoint_weight_path: Filepath where weights will be checkpointed to
            during training. This file will be rewritten by the function.
        verbose: Verbosity flag.

    # Returns:
        F1 score of the trained model
    """
    total_f1 = 0
    nb_iter = nb_classes if nb_classes > 2 else 1

    # Unpack args
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    # Save and reload initial weights after running for
    # each class to avoid learning across classes
    model.save_weights(init_weight_path)
    for i in range(nb_iter):
        if verbose:
            print('Iteration number {}/{}'.format(i + 1, nb_iter))

        model.load_weights(init_weight_path, by_name=False)
        y_train_new, y_val_new, y_test_new = prepare_labels(y_train, y_val,
                                                            y_test, i, nb_classes)
        train_gen, X_val_resamp, y_val_resamp = \
            prepare_generators(X_train, y_train_new, X_val, y_val_new,
                               batch_size, epoch_size)

        if verbose:
            print("Training..")
        callbacks = finetuning_callbacks(checkpoint_weight_path, patience, verbose=2)
        steps = int(epoch_size / batch_size)
        model.fit_generator(train_gen, steps_per_epoch=steps,
                            max_q_size=2, epochs=nb_epochs,
                            validation_data=(X_val_resamp, y_val_resamp),
                            callbacks=callbacks, verbose=0)

        # Reload the best weights found to avoid overfitting
        # Wait a bit to allow proper closing of weights file
        sleep(1)
        model.load_weights(checkpoint_weight_path, by_name=False)

        # Evaluate
        y_pred_val = np.array(model.predict(X_val, batch_size=batch_size))
        y_pred_test = np.array(model.predict(X_test, batch_size=batch_size))

        f1_test, best_t = find_f1_threshold(y_val_new, y_pred_val,
                                            y_test_new, y_pred_test)
        if verbose:
            print('f1_test: {}'.format(f1_test))
            print('best_t:  {}'.format(best_t))
        total_f1 += f1_test

    return total_f1 / nb_iter


def class_avg_chainthaw(model, nb_classes, train, val, test, batch_size,
                        loss, epoch_size, nb_epochs, checkpoint_weight_path,
                        f1_init_weight_path, patience=5,
                        initial_lr=0.001, next_lr=0.0001,
                        seed=None, verbose=True):
    """ Finetunes given model using chain-thaw and evaluates using F1.
        For a dataset with multiple classes, the model is trained once for
        each class, relabeling those classes into a binary classification task.
        The result is an average of all F1 scores for each class.

    # Arguments:
        model: Model to be finetuned.
        nb_classes: Number of classes in the given dataset.
        train: Training data, given as a tuple of (inputs, outputs)
        val: Validation data, given as a tuple of (inputs, outputs)
        test: Testing data, given as a tuple of (inputs, outputs)
        batch_size: Batch size.
        loss: Loss function to be used during training.
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        checkpoint_weight_path: Filepath where weights will be checkpointed to
            during training. This file will be rewritten by the function.
        f1_init_weight_path: Filepath where weights will be saved to and
            reloaded from before training each class. This ensures that
            each class is trained independently. This file will be rewritten.
        initial_lr: Initial learning rate. Will only be used for the first
            training step (i.e. the softmax layer)
        next_lr: Learning rate for every subsequent step.
        seed: Random number generator seed.
        verbose: Verbosity flag.

    # Returns:
        Averaged F1 score.
    """

    # Unpack args
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    total_f1 = 0
    nb_iter = nb_classes if nb_classes > 2 else 1

    model.save_weights(f1_init_weight_path)

    for i in range(nb_iter):
        if verbose:
            print('Iteration number {}/{}'.format(i + 1, nb_iter))

        model.load_weights(f1_init_weight_path, by_name=False)
        y_train_new, y_val_new, y_test_new = prepare_labels(y_train, y_val,
                                                            y_test, i, nb_classes)
        train_gen, X_val_resamp, y_val_resamp = \
            prepare_generators(X_train, y_train_new, X_val, y_val_new,
                               batch_size, epoch_size)

        if verbose:
            print("Training..")
        callbacks = finetuning_callbacks(checkpoint_weight_path, patience=patience, verbose=2)

        # Train using chain-thaw
        train_by_chain_thaw(model=model, train_gen=train_gen,
                            val_data=(X_val_resamp, y_val_resamp),
                            loss=loss, callbacks=callbacks,
                            epoch_size=epoch_size, nb_epochs=nb_epochs,
                            checkpoint_weight_path=checkpoint_weight_path,
                            initial_lr=initial_lr, next_lr=next_lr,
                            batch_size=batch_size, verbose=verbose)

        # Evaluate
        y_pred_val = np.array(model.predict(X_val, batch_size=batch_size))
        y_pred_test = np.array(model.predict(X_test, batch_size=batch_size))

        f1_test, best_t = find_f1_threshold(y_val_new, y_pred_val,
                                            y_test_new, y_pred_test)

        if verbose:
            print('f1_test: {}'.format(f1_test))
            print('best_t:  {}'.format(best_t))
        total_f1 += f1_test

    return total_f1 / nb_iter
