import sys
import os
import pickle
from .architecture import get_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
sys.path.insert(0, '..')


def model_compile_kwargs():
    return {
            "loss": "sparse_categorical_crossentropy",
            "optimizer": keras.optimizers.SGD(lr=0.01),
            "metrics": ['accuracy'],
            }


def train(iterator):
    os.makedirs(iterator.network_dir(), exist_ok=True)

    model = get_model(iterator)
    model.compile(**model_compile_kwargs())

    cback=keras.callbacks.ModelCheckpoint(iterator.weights_path(), monitor="val_accuracy", save_best_only=True)

    if iterator.use_generator:
        if iterator.dataset=="CIFAR-10" or iterator.dataset=="binarynet":
            horizontal_flip=True
        if iterator.dataset=="SVHN" or iterator.dataset=="binarynet-svhn":
            horizontal_flip=False

        datagen = ImageDataGenerator(
                width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=horizontal_flip)  # randomly flip images
        if keras.__version__[0]=='2':
            history=model.fit_generator(
                    datagen.flow(iterator.x_train, iterator.y_train, batch_size=iterator.training.batch_size),
                    steps_per_epoch=iterator.x_train.shape[0]/iterator.training.batch_size,
                    nb_epoch=iterator.training.num_epochs, validation_split=0.1,
                    verbose=2, callbacks=[cback])
        if keras.__version__[0]=='1':
            history=model.fit_generator(
                    datagen.flow(iterator.x_train, iterator.y_train, batch_size=iterator.training.batch_size),
                    samples_per_epoch=iterator.x_train.shape[0], nb_epoch=iterator.training.num_epochs,
                    verbose=2, validation_split=0.1, callbacks=[cback])
    else:
        if keras.__version__[0]=='2':
            history=model.fit(iterator.x_train, iterator.y_train, batch_size=iterator.training.batch_size,validation_split=0.1, verbose=2, epochs=iterator.training.num_epochs,callbacks=[cback])
        if keras.__version__[0]=='1':
            history=model.fit(iterator.x_train, iterator.y_train, batch_size=iterator.training.batch_size,validation_split=0.1, verbose=2, nb_epoch=iterator.training.num_epochs,callbacks=[cback])

    dic={'hard': history.history}
    with open(iterator.history_path(), "wb") as handle:
        pickle.dump(dic, handle)


def infer(iterator):
    os.makedirs(iterator.inference_repeat_dir(), exist_ok=True)

    model = get_model(iterator)
    model.load_weights(iterator.weights_path())
    model.compile(**model_compile_kwargs())
    # All at once, so that a single value of average power is computed for each synaptic layer.
    score = model.evaluate(iterator.x_test, iterator.y_test, verbose=0, batch_size=iterator.x_test.shape[0])

    print("Test loss: %0.4f\nTest accuracy: %0.4f"%(score[0], score[1]))

    loss_path = iterator.loss_path()
    open(loss_path, "a").close()
    tf.print(score[0], output_stream="file://{}".format(loss_path))

    accuracy_path = iterator.accuracy_path()
    open(accuracy_path, "a").close()
    tf.print(score[1], output_stream="file://{}".format(accuracy_path))
