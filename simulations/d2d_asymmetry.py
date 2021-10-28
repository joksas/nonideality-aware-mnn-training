from training import callbacks
from training.iterator import Inference, Iterator, Training

from . import devices, utils

DATASET = "mnist"


def custom_iterator(training_setup, inference_setups):
    inferences = [
        Inference(**utils.get_inference_params(), **setup) for setup in inference_setups
    ]
    training = Training(
        **utils.get_training_params(), is_regularized=False, **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_nonideal_iterators():
    return [
        custom_iterator(devices.symmetric_d2d(), [devices.symmetric_d2d()]),
        custom_iterator(devices.asymmetric_d2d(), [devices.asymmetric_d2d()]),
    ]


def get_iterators():
    return get_nonideal_iterators()


def main():
    for iterator in get_nonideal_iterators():
        iterator.train(use_test_callback=True)
        iterator.infer()
