import os
import pickle

import tensorflow as tf

import awarememristor.training.callbacks as callbacks_
from awarememristor.training.architecture import get_model


def train(iterator, callbacks: list[callbacks_.Callback] = []) -> None:
    """Trains using iterator settings and saves information once done."""
    os.makedirs(iterator.weights_dir(), exist_ok=True)

    validation_data = None
    num_checkpoint_callbacks = 0
    for callback in callbacks:
        if isinstance(callback, callbacks_.Checkpoint):
            num_checkpoint_callbacks += 1
            if isinstance(callback, callbacks_.StandardCheckpoint):
                validation_data = iterator.data("validation")

    if num_checkpoint_callbacks != 1:
        raise ValueError("One checkpoint callback must be supplied during training!")

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
        if isinstance(callback, callbacks_.MemristiveCallback):
            if callback.name() in info["callback_infos"]:
                raise KeyError(f'Callback "{callback.name()}" already exists!')
            info["callback_infos"][callback.name()] = callback.info()

    with open(iterator.info_path(), "wb") as handle:
        pickle.dump(info, handle)


def infer(iterator):
    """Performs inference using iterator settings and saves metrics to separate files."""
    os.makedirs(iterator.inference_repeat_dir(), exist_ok=True)

    model = get_model(iterator)

    score = model.evaluate(iterator.data("testing"), verbose=0)

    print(f"Test loss: {score[0]:.4f}\nTest accuracy: {score[1]:.4f}")

    for var, path in zip(score, [iterator.loss_path(), iterator.accuracy_path()]):
        with open(path, mode="a", encoding="utf-8"):
            tf.print(var, output_stream=f"file://{path}")
