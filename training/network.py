import sys
import os
import pickle
from .architecture import get_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.insert(0, "..")


def train(iterator):
    os.makedirs(iterator.network_dir(), exist_ok=True)

    model = get_model(iterator)

    cback=keras.callbacks.ModelCheckpoint(
            iterator.weights_path(),
            monitor="val_accuracy",
            save_best_only=True)

    validation_split = 0.1
    verbose = 2

    if iterator.use_generator:
        datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                validation_split=validation_split,
                )

        history = model.fit(
                datagen.flow(iterator.x_train, iterator.y_train, batch_size=iterator.training.batch_size, subset="training"),
                validation_data=datagen.flow(iterator.x_train, iterator.y_train, subset="validation"),
                epochs=iterator.training.num_epochs,
                callbacks=[cback],
                )
    else:
        history=model.fit(
                iterator.x_train, iterator.y_train,
                batch_size=iterator.training.batch_size,
                validation_split=validation_split,
                verbose=verbose,
                epochs=iterator.training.num_epochs,
                callbacks=[cback])

    info = {
            "history": history.history,
            "validation_split": validation_split,
            "batch_size": iterator.training.batch_size,
            }

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
