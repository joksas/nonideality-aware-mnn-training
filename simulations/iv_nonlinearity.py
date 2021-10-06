from training.iterator import Iterator, Training, IVNonlinearity


DATASET = "MNIST"
NUM_EPOCHS = 1000
BATCH_SIZE = 32
NUM_TRAINING_REPEATS = 1
DEVICE_TYPES = [
        [1/1003, 1/284.6, IVNonlinearity(2.132, 0.095)],
        [1/1295000, 1/366200, IVNonlinearity(2.989, 0.369)],
        ]


def custom_iterator(G_min, G_max, nonidealities, is_regularized):
    return Iterator(
            DATASET,
            G_min,
            G_max,
            Training(num_repeats=NUM_TRAINING_REPEATS, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                nonidealities=nonidealities,
                is_regularized=is_regularized,
                ),
            )


def get_iterators():
    iterators = []

    # Without I-V nonlinearity and not regularized.
    iterator = custom_iterator(None, None, {}, False)
    iterators.append(iterator)

    for G_min, G_max, iv_nonlinearity in DEVICE_TYPES:
        # With I-V nonlinearity and not regularized.
        iterator = custom_iterator(G_min, G_max, {"iv_nonlinearity": iv_nonlinearity}, False)
        iterators.append(iterator)

        # With I-V nonlinearity and regularized.
        iterator = custom_iterator(G_min, G_max, {"iv_nonlinearity": iv_nonlinearity}, True)
        iterators.append(iterator)

    return iterators


def main():
    for iterator in get_iterators():
        iterator.train()


if __name__ == "__main__":
    main()
