from training import callbacks
from training.iterator import Inference, Iterator, Training

from . import (checkpoint_comparison, d2d_asymmetry, devices, iv_nonlinearity,
               iv_nonlinearity_and_stuck, iv_nonlinearity_cnn,
               iv_nonlinearity_cnn_checkpoint_frequency, utils)


def custom_iterator(training_setup, inference_setups, dataset):
    inferences = [
        Inference(utils.get_inference_params(), **setup) for setup in inference_setups
    ]
    training = Training(
        utils.get_training_params(), is_regularized=False, **training_setup
    )

    return Iterator(dataset, training, inferences)


def get_mnist_iterator():
    iterators = [
        iv_nonlinearity.get_ideal_iterator(),
        iv_nonlinearity_and_stuck.get_ideal_iterator(),
    ]

    return custom_iterator(
        iterators[0].training,
        [inference for inference in iterator.inferences for iterator in iterators],
        "mnist",
    )


def get_cifar10_iterator():
    iterators = [
        iv_nonlinearity_cnn.get_ideal_iterator(),
    ]

    return custom_iterator(
        iterators[0].training,
        [inference for inference in iterator.inferences for iterator in iterators],
        "cifar10",
    )


def get_iterators():
    return [
        get_mnist_iterator(),
        get_cifar10_iterator(),
    ]


def main():
    for iterator in get_iterators():
        iterator.train(use_test_callback=True)
        iterator.infer()
