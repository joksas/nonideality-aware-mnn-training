import sys
import os
import pickle
import copy
from .architecture import get_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.insert(0, "..")


def train(iterator, callbacks=[]):
    os.makedirs(iterator.network_dir(), exist_ok=True)

    callback_names = [callback.name() for callback in callbacks]
    if sum(x in ["regular_checkpoint", "memristive_checkpoint"] for x in callback_names) != 1:
        raise ValueError("One checkpoint callback must be supplied during training!")
    validation_freq = 1
    if "memristive_checkpoint" in callback_names:
        # Arbitrarily large number to avoid evaluating using validation set. We still want to use it
        # for access through callbacks though.
        validation_freq = 1000000

    model = get_model(iterator)

    if iterator.use_generator:
        datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                validation_split=iterator.training.validation_split,
                )

        history = model.fit(
                datagen.flow(iterator.x_train, iterator.y_train, batch_size=iterator.training.batch_size, subset="training"),
                validation_data=datagen.flow(iterator.x_train, iterator.y_train, subset="validation"),
                epochs=iterator.training.num_epochs,
                callbacks=callbacks,
                verbose=2,
                validation_freq=validation_freq,
                )
    else:
        history=model.fit(
                iterator.x_train, iterator.y_train,
                batch_size=iterator.training.batch_size,
                validation_split=iterator.training.validation_split,
                verbose=2,
                epochs=iterator.training.num_epochs,
                callbacks=callbacks,
                validation_freq=validation_freq,
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
                raise KeyError(f"Callback \"{callback.name()}\" already exists!")
            info["callback_infos"][callback.name()] = callback.info()
        except AttributeError:
            pass

    with open(iterator.info_path(), "wb") as handle:
        pickle.dump(info, handle)


def infer(iterator):
    os.makedirs(iterator.inference_repeat_dir(), exist_ok=True)

    model = get_model(iterator)

    score = model.evaluate(iterator.x_test, iterator.y_test, verbose=0, batch_size=10)

    print("Test loss: %0.4f\nTest accuracy: %0.4f"%(score[0], score[1]))

    loss_path = iterator.loss_path()
    open(loss_path, "a").close()
    tf.print(score[0], output_stream="file://{}".format(loss_path))

    accuracy_path = iterator.accuracy_path()
    open(accuracy_path, "a").close()
    tf.print(score[1], output_stream="file://{}".format(accuracy_path))


