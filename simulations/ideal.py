from training import callbacks
from training.iterator import Inference, Iterator, Training

from . import (checkpoint_comparison, d2d_asymmetry, devices, iv_nonlinearity,
               iv_nonlinearity_and_stuck, iv_nonlinearity_cnn,
               iv_nonlinearity_cnn_checkpoint_frequency, utils)


def get_mnist_iterator():
    iterators = [
        iv_nonlinearity.get_ideal_iterator(),
        iv_nonlinearity_and_stuck.get_ideal_iterator(),
    ]

    return Iterator(
        "mnist",
        iterators[0].training,
        [inference for iterator in iterators for inference in iterator.inferences],
    )


def get_cifar10_iterator():
    iterators = [
        iv_nonlinearity_cnn.get_ideal_iterator(),
    ]

    return Iterator(
        "cifar10",
        iterators[0].training,
        [inference for iterator in iterators for inference in iterator.inferences],
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
