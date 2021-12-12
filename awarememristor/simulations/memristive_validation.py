from awarememristor.simulations import devices, utils
from awarememristor.training.iterator import Inference, Iterator, Training

DATASET = "mnist"


def custom_iterator(
    training_setup, inference_setups, use_combined_validation, num_training_repeats: int = None
):
    training_params = utils.get_training_params()
    if num_training_repeats is not None:
        training_params["num_repeats"] = num_training_repeats
    inferences = [Inference(**utils.get_inference_params(), **setup) for setup in inference_setups]
    training = Training(
        **training_params,
        is_regularized=False,
        use_combined_validation=use_combined_validation,
        **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_ideal_iterator():
    return custom_iterator(devices.ideal(), [devices.high_magnitude_more_uniform_d2d()], False)


def get_nonideal_iterators():
    return [
        custom_iterator(
            devices.high_magnitude_more_uniform_d2d(),
            [devices.high_magnitude_more_uniform_d2d()],
            True,
            # Validation is utilized during training, so to evaluate the
            # effectiveness of different methods, we need to increase the
            # sample size of trained networks.
            num_training_repeats=100,
        ),
    ]


def get_iterators():
    return [
        get_ideal_iterator(),
        *get_nonideal_iterators(),
    ]


def main():
    for iterator in get_nonideal_iterators():
        iterator.train(use_test_callback=True)
        iterator.training.is_standard_validation_mode = False
        iterator.infer()
        iterator.training.is_standard_validation_mode = True
        iterator.infer()
