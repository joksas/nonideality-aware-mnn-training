import os

from training import callbacks
from training.iterator import Inference, Iterator, Training

from . import (d2d_asymmetry, devices, ideal, iv_nonlinearity,
               iv_nonlinearity_and_stuck, stuck_low, utils)

DATASET = "mnist"
INFERENCE_SETUPS = [
    devices.ideal(),
    devices.low_R(),
    devices.high_R(),
    devices.stuck_low(),
    devices.high_R_and_stuck(),
    devices.symmetric_d2d(),
    devices.asymmetric_d2d(),
]


def custom_iterator(training_setup, inference_setups, is_regularized):
    inferences = [Inference(**utils.get_inference_params(), **setup) for setup in inference_setups]
    training = Training(
        **utils.get_training_params(), is_regularized=is_regularized, **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_iterators():
    iterators = [
        ideal.get_mnist_iterator(),
        *iv_nonlinearity.get_nonideal_iterators(),
        *iv_nonlinearity_and_stuck.get_nonideal_iterators(),
        *stuck_low.get_nonideal_iterators(),
        *d2d_asymmetry.get_nonideal_iterators()[:2],
    ]
    inferences = [
        Inference(**utils.get_inference_params(), **inference_setup)
        for inference_setup in INFERENCE_SETUPS
    ]

    for idx, iterator in enumerate(iterators):
        for inference in inferences:
            if inference not in iterator.inferences:
                iterators[idx].inferences.append(inference)

    return iterators


def main():
    for iterator in get_iterators():
        iterator.infer()
