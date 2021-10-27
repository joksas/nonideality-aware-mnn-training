from training import callbacks
from training.iterator import Inference, Iterator, Training

from . import devices

DATASET = "cifar10"
NUM_EPOCHS = 1000
BATCH_SIZE = 64
NUM_TRAINING_REPEATS = 5
NUM_INFERENCE_REPEATS = 25


def custom_iterator(training_setup, inference_setups):
    inferences = [
        Inference(num_repeats=NUM_INFERENCE_REPEATS, **setup)
        for setup in inference_setups
    ]
    training = Training(
        num_repeats=NUM_TRAINING_REPEATS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        is_regularized=False,
        memristive_validation_freq=5,
        **training_setup
    )

    return Iterator(DATASET, training, inferences)


def get_iterators():
    iterators = [
        custom_iterator(devices.high_R(), [devices.high_R()]),
    ]

    return iterators


def main():
    for iterator in get_iterators():
        iterator.train(use_test_callback=True)
        iterator.infer()
