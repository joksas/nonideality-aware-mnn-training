from awarememristor.training import callbacks
from awarememristor.training.iterator import Inference, Iterator, Training

from . import devices, iv_nonlinearity_cnn, utils

DATASET = "cifar10"


def custom_iterator(training_setup, inference_setups, memristive_validation_freq=None):
    inferences = [Inference(**utils.get_inference_params(), **setup) for setup in inference_setups]
    training = Training(
        **utils.get_training_params(),
        is_regularized=False,
        memristive_validation_freq=memristive_validation_freq,
        **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_nonideal_iterators():
    return [custom_iterator(devices.high_R(), [devices.high_R()], memristive_validation_freq=5)]


def get_iterators():
    return [
        iv_nonlinearity_cnn.get_ideal_iterator(),
        *get_nonideal_iterators(),
    ]


def main():
    for iterator in get_nonideal_iterators():
        iterator.train(use_test_callback=True)
        iterator.infer()
