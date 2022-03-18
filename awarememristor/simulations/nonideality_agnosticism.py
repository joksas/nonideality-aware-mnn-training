from awarememristor.simulations import (devices, high_d2d, ideal,
                                        iv_nonlinearity,
                                        iv_nonlinearity_and_stuck_on,
                                        stuck_distribution, stuck_off, utils,
                                        weight_implementation)
from awarememristor.training.iterator import Inference, Iterator, Training

DATASET = "mnist"


def custom_iterator(training_setup, inference_setups, is_regularized):
    inferences = [Inference(**utils.get_inference_params(), **setup) for setup in inference_setups]
    training = Training(
        **utils.get_training_params(), is_regularized=is_regularized, **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_iterators():
    inference_setups = [
        devices.ideal(),
        devices.SiO_x(False),
        devices.SiO_x(True),
        devices.stuck_off(),
        devices.SiO_x_high_nonlinearity_and_stuck_on(),
        devices.more_uniform_d2d(),
        devices.less_uniform_d2d(),
        devices.high_magnitude_more_uniform_d2d(),
        devices.Ta_HfO2(),
    ]

    iterators = [
        ideal.get_mnist_iterator(),
        *iv_nonlinearity.get_nonideal_iterators(),
        *iv_nonlinearity_and_stuck_on.get_nonideal_iterators(),
        *stuck_off.get_nonideal_iterators(),
        *weight_implementation.get_nonideal_iterators()[-4:],
        *stuck_distribution.get_nonideal_iterators(),
        *high_d2d.get_nonideal_iterators(),
    ]
    inferences = [
        Inference(**utils.get_inference_params(), **inference_setup)
        for inference_setup in inference_setups
    ]

    for idx, iterator in enumerate(iterators):
        # Use the same number of repeats for all training setups.
        iterator.training.num_repeats = utils.get_training_params()["num_repeats"]
        for inference in inferences:
            if inference not in iterator.inferences:
                iterators[idx].inferences.append(inference)

    return iterators


def main():
    for iterator in get_iterators():
        iterator.infer()
