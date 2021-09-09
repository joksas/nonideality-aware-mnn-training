import sys
import os
import pickle
from .architecture import get_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.insert(0, "..")


def model_compile_kwargs(dataset):
    if dataset == "MNIST":
        return  {
                "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
                "optimizer": keras.optimizers.SGD(),
                "metrics": ["accuracy"],
                }
    elif dataset == "CIFAR-10":
        return {
                "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
                "optimizer": keras.optimizers.Adam(),
                "metrics": ["accuracy"]
                }


def train(iterator):
    os.makedirs(iterator.network_dir(), exist_ok=True)

    model = get_model(iterator)
    model.compile(**model_compile_kwargs(iterator.dataset))

    cback=keras.callbacks.ModelCheckpoint(
            iterator.weights_path(),
            monitor="val_accuracy",
            save_best_only=True)

    validation_split = 0.1
    verbose = 2
    history=model.fit(
            iterator.x_train, iterator.y_train,
            batch_size=iterator.training.batch_size,
            validation_split=validation_split,
            verbose=verbose,
            epochs=iterator.training.num_epochs,
            callbacks=[cback])

    dic={"hard": history.history}
    with open(iterator.history_path(), "wb") as handle:
        pickle.dump(dic, handle)


def infer(iterator):
    os.makedirs(iterator.inference_repeat_dir(), exist_ok=True)

    model = get_model(iterator)
    model.load_weights(iterator.weights_path())
    model.compile(**model_compile_kwargs(iterator.dataset))
    # All at once, so that a single value of average power is computed for each synaptic layer.
    score = model.evaluate(iterator.x_test, iterator.y_test, verbose=0, batch_size=iterator.x_test.shape[0])

    print("Test loss: %0.4f\nTest accuracy: %0.4f"%(score[0], score[1]))

    loss_path = iterator.loss_path()
    open(loss_path, "a").close()
    tf.print(score[0], output_stream="file://{}".format(loss_path))

    accuracy_path = iterator.accuracy_path()
    open(accuracy_path, "a").close()
    tf.print(score[1], output_stream="file://{}".format(accuracy_path))
