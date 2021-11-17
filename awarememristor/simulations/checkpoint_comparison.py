from awarememristor.training import callbacks
from awarememristor.training.iterator import Inference, Iterator, Training

from . import devices, utils

DATASET = "mnist"


def custom_iterator(training_setup, inference_setups, force_regular):
    inferences = [Inference(**utils.get_inference_params(), **setup) for setup in inference_setups]
    training = Training(
        **utils.get_training_params(),
        is_regularized=False,
        force_regular_checkpoint=force_regular,
        **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_nonideal_iterators():
    return [
        custom_iterator(
            devices.high_magnitude_more_uniform_d2d(),
            [devices.high_magnitude_more_uniform_d2d()],
            True,
        ),
        custom_iterator(
            devices.high_magnitude_more_uniform_d2d(),
            [devices.high_magnitude_more_uniform_d2d()],
            False,
        ),
    ]


def get_iterators():
    return get_nonideal_iterators()


def main():
    for iterator in get_nonideal_iterators():
        iterator.train(use_test_callback=True)
        iterator.infer()
