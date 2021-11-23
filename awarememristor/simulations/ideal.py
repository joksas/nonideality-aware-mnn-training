from awarememristor.simulations import (checkpoint_comparison,
                                        differential_pair_separation,
                                        iv_nonlinearity,
                                        iv_nonlinearity_and_stuck_on,
                                        iv_nonlinearity_cnn,
                                        stuck_distribution, stuck_off)
from awarememristor.training import callbacks
from awarememristor.training.iterator import Inference, Iterator, Training


def get_mnist_iterator():
    iterators = [
        iv_nonlinearity.get_ideal_iterator(),
        stuck_off.get_ideal_iterator(),
        iv_nonlinearity_and_stuck_on.get_ideal_iterator(),
        stuck_distribution.get_ideal_iterator(),
        differential_pair_separation.get_ideal_iterator(),
        checkpoint_comparison.get_ideal_iterator(),
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
