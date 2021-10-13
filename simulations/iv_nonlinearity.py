from training.iterator import Iterator, Training, Inference
from training import callbacks
from . import devices


DATASET = "MNIST"
NUM_EPOCHS = 1000
BATCH_SIZE = 32
NUM_TRAINING_REPEATS = 5
NUM_INFERENCE_REPEATS = 25


def custom_iterator(training_setup, inference_setups, is_regularized):
    inferences = [Inference(num_repeats=NUM_INFERENCE_REPEATS, **setup) for setup in inference_setups]
    training = Training(num_repeats=NUM_TRAINING_REPEATS, num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE, is_regularized=is_regularized, **training_setup)

    return Iterator(DATASET, training, inferences)


def get_iterators():
    iterators = [
            custom_iterator(devices.ideal(), [devices.low_R(), devices.high_R()], False),
            custom_iterator(devices.low_R(), [devices.low_R()], False),
            custom_iterator(devices.low_R(), [devices.low_R()], True),
            custom_iterator(devices.high_R(), [devices.high_R()], False),
            custom_iterator(devices.high_R(), [devices.high_R()], True),
            ]

    return iterators


def main():
    for iterator in get_iterators():
        iterator.train(callbacks=[callbacks.TestCallback(iterator)])
        iterator.infer()
