from training.iterator import Iterator, Training, Inference
from training import callbacks
from . import devices


DATASET = "mnist"
NUM_EPOCHS = 1000
BATCH_SIZE = 64
NUM_TRAINING_REPEATS = 5
NUM_INFERENCE_REPEATS = 25


def custom_iterator(training_setup, inference_setups):
    inferences = [Inference(num_repeats=NUM_INFERENCE_REPEATS, **setup) for setup in inference_setups]
    training = Training(num_repeats=NUM_TRAINING_REPEATS, num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE, is_regularized=False, force_regular_checkpoint=True, **training_setup)

    return Iterator(DATASET, training, inferences)


def get_iterators():
    iterators = [
            custom_iterator(devices.asymmetric_d2d(), [devices.asymmetric_d2d()]),
            ]

    return iterators


def main():
    for iterator in get_iterators():
        iterator.train(use_test_callback=True)
        iterator.infer()