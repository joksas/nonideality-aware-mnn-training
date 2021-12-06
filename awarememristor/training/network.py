import os
import pickle
import sys

import tensorflow as tf

from awarememristor.training.architecture import get_model

sys.path.insert(0, "..")


def train(iterator, callbacks=[]):
    os.makedirs(iterator.network_dir(), exist_ok=True)

    callback_names = [callback.name() for callback in callbacks]
    if sum(x in ["standard_checkpoint", "memristive_checkpoint"] for x in callback_names) != 1:
        raise ValueError("One checkpoint callback must be supplied during training!")

    validation_data = None
    if "standard_checkpoint" in callback_names:
        validation_data = iterator.data("validation")

    model = get_model(iterator)

    history = model.fit(
        iterator.data("training"),
        validation_data=validation_data,
        verbose=2,
        epochs=iterator.training.num_epochs,
        callbacks=callbacks,
    )

    info = {
        "history": history.history,
        "validation_split": iterator.training.validation_split,
        "batch_size": iterator.training.batch_size,
        "callback_infos": {},
    }
    for callback in callbacks:
        try:
            if callback.name() in info["callback_infos"]:
                raise KeyError(f'Callback "{callback.name()}" already exists!')
            info["callback_infos"][callback.name()] = callback.info()
        except AttributeError:
            pass

    with open(iterator.info_path(), "wb") as handle:
        pickle.dump(info, handle)


def infer(iterator):
    os.makedirs(iterator.inference_repeat_dir(), exist_ok=True)

    model = get_model(iterator)

    score = model.evaluate(iterator.data("testing"), verbose=0)

    print(f"Test loss: {score[0]:.4f}\nTest accuracy: {score[1]:.4f}")

    for var, path in zip(score, [iterator.loss_path(), iterator.accuracy_path()]):
        with open(path, mode="a", encoding="utf-8"):
            tf.print(var, output_stream=f"file://{path}")
