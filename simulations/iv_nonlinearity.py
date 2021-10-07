from training.iterator import Iterator, Training, IVNonlinearity, TrainingCallback, Inference


DATASET = "MNIST"
NUM_EPOCHS = 4
BATCH_SIZE = 32
NUM_TRAINING_REPEATS = 1
NUM_INFERENCE_REPEATS = 3
IDEAL = {
        "G_min": None,
        "G_max": None,
        "nonidealities": {}
        }
LOW_R_DEVICE = {
        "G_min": 1/1003,
        "G_max": 1/284.6,
        "nonidealities": {"iv_nonlinearity": IVNonlinearity(2.132, 0.095)}
        }
HIGH_R_DEVICE = {
        "G_min": 1/1295000,
        "G_max": 1/366200,
        "nonidealities": {"iv_nonlinearity": IVNonlinearity(2.989, 0.369)}
        }


def custom_iterator(training_setup, inference_setups, is_regularized):
    inferences = [Inference(num_repeats=NUM_INFERENCE_REPEATS, **setup) for setup in inference_setups]
    training = Training(num_repeats=NUM_TRAINING_REPEATS, num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE, is_regularized=is_regularized, **training_setup)

    return Iterator(DATASET, training, inferences)


def get_iterators():
    iterators = [
            custom_iterator(IDEAL, [LOW_R_DEVICE, HIGH_R_DEVICE], False),
            custom_iterator(LOW_R_DEVICE, [LOW_R_DEVICE], False),
            custom_iterator(LOW_R_DEVICE, [LOW_R_DEVICE], True),
            custom_iterator(HIGH_R_DEVICE, [HIGH_R_DEVICE], False),
            custom_iterator(HIGH_R_DEVICE, [HIGH_R_DEVICE], True),
            ]

    return iterators


def main():
    for iterator in get_iterators():
        iterator.train([TrainingCallback(iterator)])
        iterator.infer()


if __name__ == "__main__":
    main()
