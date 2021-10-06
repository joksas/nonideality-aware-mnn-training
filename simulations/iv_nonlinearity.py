from training.iterator import Iterator, Training, IVNonlinearity, TrainingCallback


DATASET = "MNIST"
NUM_EPOCHS = 1000
BATCH_SIZE = 32
NUM_TRAINING_REPEATS = 1
LOW_R_DEVICE = [1/1003, 1/284.6, {"iv_nonlinearity": IVNonlinearity(2.132, 0.095)}]
HIGH_R_DEVICE = [1/1295000, 1/366200, {"iv_nonlinearity": IVNonlinearity(2.989, 0.369)}]


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
    callbacks_lst = []

    # Without I-V nonlinearity and not regularized.
    iterator = custom_iterator(None, None, {}, False)
    iterators.append(iterator)
    callbacks_lst.append([
        TrainingCallback(iterator, *LOW_R_DEVICE),
        TrainingCallback(iterator, *HIGH_R_DEVICE),
        ])

    for device in [LOW_R_DEVICE, HIGH_R_DEVICE]:
        for is_regularized in [False, True]:
            iterator = custom_iterator(*device, is_regularized)
            iterators.append(iterator)
            callbacks_lst.append([
                TrainingCallback(iterator, *device),
                ])

    return iterators, callbacks_lst


def main():
    for iterator, callbacks in zip(*get_iterators()):
        iterator.train(callbacks=callbacks)
        print(iterator.info())


if __name__ == "__main__":
    main()
