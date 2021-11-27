from awarememristor.simulations import devices, utils
from awarememristor.training.iterator import Inference, Iterator, Training

DATASET = "mnist"


def custom_iterator(
    training_setup,
    inference_setups,
    is_regularized=False,
    force_standard_w=False,
    mapping_rule="default",
):
    inferences = [
        Inference(mapping_rule=mapping_rule, **utils.get_inference_params(), **setup)
        for setup in inference_setups
    ]
    training = Training(
        **utils.get_training_params(),
        is_regularized=is_regularized,
        force_standard_w=force_standard_w,
        mapping_rule=mapping_rule,
        **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_ideal_iterator():
    iterator = custom_iterator(
        devices.ideal(), [devices.more_uniform_d2d(), devices.less_uniform_d2d()], False
    )

    return iterator


def get_nonideal_iterators():
    return [
        custom_iterator(
            devices.more_uniform_d2d(),
            [devices.more_uniform_d2d()],
            force_standard_w=True,
        ),
        custom_iterator(
            devices.more_uniform_d2d(),
            [devices.more_uniform_d2d()],
            force_standard_w=True,
            mapping_rule="avg",
        ),
        custom_iterator(
            devices.less_uniform_d2d(),
            [devices.less_uniform_d2d()],
            force_standard_w=True,
        ),
        custom_iterator(
            devices.less_uniform_d2d(),
            [devices.less_uniform_d2d()],
            force_standard_w=True,
            mapping_rule="avg",
        ),
        custom_iterator(devices.more_uniform_d2d(), [devices.more_uniform_d2d()]),
        custom_iterator(
            devices.more_uniform_d2d(), [devices.more_uniform_d2d()], is_regularized=True
        ),
        custom_iterator(devices.less_uniform_d2d(), [devices.less_uniform_d2d()]),
        custom_iterator(
            devices.less_uniform_d2d(), [devices.less_uniform_d2d()], is_regularized=True
        ),
    ]


def get_iterators():
    return [
        *get_nonideal_iterators(),
    ]


def main():
    for iterator in get_nonideal_iterators():
        iterator.train(use_test_callback=True)
        iterator.infer()
