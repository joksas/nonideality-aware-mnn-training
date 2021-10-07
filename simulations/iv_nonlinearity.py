from training.iterator import Iterator, Training, TrainingCallback, Inference
from . import devices


DATASET = "MNIST"
NUM_EPOCHS = 4
BATCH_SIZE = 32
NUM_TRAINING_REPEATS = 1
NUM_INFERENCE_REPEATS = 3


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
        iterator.train([TrainingCallback(iterator)])
        iterator.infer()


if __name__ == "__main__":
    main()
