from awarememristor.simulations import devices, utils
from awarememristor.training.iterator import Inference, Iterator, Training

DATASET = "mnist"


def custom_iterator(training_setup, inference_setups, is_regularized):
    inferences = [Inference(**utils.get_inference_params(), **setup) for setup in inference_setups]
    training = Training(
        **utils.get_training_params(), is_regularized=is_regularized, **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_ideal_iterator():
    return custom_iterator(
        devices.ideal(), [devices.SiO_x_pf(False), devices.SiO_x_pf(True)], False
    )


def get_nonideal_iterators():
    return [
        custom_iterator(devices.SiO_x_pf(False), [devices.SiO_x_pf(False)], False),
        custom_iterator(devices.SiO_x_pf(False), [devices.SiO_x_pf(False)], True),
        custom_iterator(devices.SiO_x_pf(True), [devices.SiO_x_pf(True)], False),
        custom_iterator(devices.SiO_x_pf(True), [devices.SiO_x_pf(True)], True),
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
