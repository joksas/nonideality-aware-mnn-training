from training import callbacks
from training.iterator import Inference, Iterator, Training

from . import devices, utils

DATASET = "mnist"


def custom_iterator(training_setup, inference_setups, is_regularized):
    inferences = [
        Inference(**utils.get_inference_params(), **setup) for setup in inference_setups
    ]
    training = Training(
        **utils.get_training_params(), is_regularized=is_regularized, **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_ideal_iterator():
    return custom_iterator(devices.ideal(), [devices.low_R(), devices.high_R()], False)


def get_nonideal_iterators():
    return [
        custom_iterator(devices.low_R(), [devices.low_R()], False),
        custom_iterator(devices.low_R(), [devices.low_R()], True),
        custom_iterator(devices.high_R(), [devices.high_R()], False),
        custom_iterator(devices.high_R(), [devices.high_R()], True),
    ]


def get_iterators():
    return [
        get_ideal_iterator(),
        *get_nonideal_iterators(),
    ]


def main():
    for iterator in get_nonideal_iterators():
        iterator.train(use_test_callback=True)
        iterator.infer()
