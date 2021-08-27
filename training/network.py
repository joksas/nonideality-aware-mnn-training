import numpy as np
import pickle
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
sys.path.insert(0, '..')
from .architecture import get_model


def train(iterator):
    os.makedirs(iterator.network_dir(), exist_ok=True)

    model = get_model(iterator)

    lr = 0.01
    opt = keras.optimizers.SGD(lr=lr)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

    cback=keras.callbacks.ModelCheckpoint(iterator.weights_path(), monitor='val_accuracy', save_best_only=True)

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
                    datagen.flow(iterator.x_train, y_train, batch_size=iterator.training.batch_size),
                    steps_per_epoch=iterator.x_train.shape[0]/iterator.training.batch_size,
                    nb_epoch=num_epochs, validation_split=0.1,
                    verbose=2, callbacks=[cback])
        if keras.__version__[0]=='1':
            history=model.fit_generator(
                    datagen.flow(iterator.x_train, y_train, batch_size=iterator.training.batch_size),
                    samples_per_epoch=iterator.x_train.shape[0], nb_epoch=num_epochs,
                    verbose=2, validation_split=0.1, callbacks=[cback])
    else:
        if keras.__version__[0]=='2':
            history=model.fit(iterator.x_train, iterator.y_train, batch_size=iterator.training.batch_size,validation_split=0.1, verbose=2, epochs=iterator.training.num_epochs,callbacks=[cback])
        if keras.__version__[0]=='1':
            history=model.fit(iterator.x_train, iterator.y_train, batch_size=iterator.training.batch_size,validation_split=0.1, verbose=2, nb_epoch=num_epochs,callbacks=[cback])

    dic={'hard': history.history}
    with open(iterator.history_path(), 'wb') as handle:
        pickle.dump(dic, handle)


def evaluate(dir_path, dataset, x_test, y_test, batch_size, group_idx=None, is_regularized=True, log_dir_full_path=None):
    weights_path= dir_path + "/output_model.h5"
    model = get_model(dataset, batch_size, group_idx=group_idx, is_regularized=is_regularized, log_dir_full_path=log_dir_full_path)
    model.load_weights(weights_path)
    opt = keras.optimizers.SGD()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    if log_dir_full_path is None:
        print("Test loss was %0.4f, test accuracy was %0.4f"%(score[0], score[1]))
    else:
        log_file_full_path = "{}/accuracy.csv".format(log_dir_full_path)
        open(log_file_full_path, "a").close()
        tf.print(score[1], output_stream="file://{}".format(log_file_full_path))

