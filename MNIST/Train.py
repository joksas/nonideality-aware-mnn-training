import numpy as np
import pickle
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
sys.path.insert(0, '..')
from model_architectures import get_model


def load_svhn(path_to_dataset):
    import scipy.io as sio
    train = sio.loadmat(path_to_dataset+'/train.mat')
    test = sio.loadmat(path_to_dataset+'/test.mat')
    extra = sio.loadmat(path_to_dataset+'/extra.mat')
    x_train = np.transpose(train['X'], [3, 0, 1, 2])
    y_train = train['y']-1

    x_test = np.transpose(test['X'], [3, 0, 1, 2])
    y_test = test['y']-1

    x_extra = np.transpose(extra['X'], [3, 0, 1, 2])
    y_extra=extra['y']-1

    x_train = np.concatenate((x_train, x_extra), axis=0)
    y_train = np.concatenate((y_train, y_extra), axis=0)

    return (x_train, y_train), (x_test, y_test)

def get_examples(dataset):
    if dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # convert class vectors to binary class matrices
        x_train = x_train.reshape(-1,784)
        x_test = x_test.reshape(-1,784)
        use_generator = False
    elif dataset == "CIFAR-10" or dataset == "binarynet":
        use_generator = True
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == "SVHN" or dataset == "binarynet-svhn":
        use_generator = True
        (x_train, y_train), (x_test, y_test) = load_svhn('./svhn_data')
    else:
        raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN, binarynet, binarynet-svhn].")

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test, use_generator

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.025
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def train_network(dir_path, dataset, x_train, y_train, num_epochs, use_generator, batch_size, group_idx=None):
    os.makedirs(dir_path, exist_ok=True)

    model = get_model(dataset, batch_size, group_idx=group_idx)

    lr = 0.01
    opt = keras.optimizers.SGD(lr=lr)
    model.compile(
            loss='sparse_categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

    weights_path = dir_path + "/output_model.h5"
    cback=keras.callbacks.ModelCheckpoint(
            weights_path, monitor='val_accuracy', save_best_only=True)

    if use_generator:
        if dataset=="CIFAR-10" or dataset=="binarynet":
            horizontal_flip=True
        if dataset=="SVHN" or dataset=="binarynet-svhn":
            horizontal_flip=False

        datagen = ImageDataGenerator(
                width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=horizontal_flip)  # randomly flip images
        if keras.__version__[0]=='2':
            history=model.fit_generator(
                    datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0]/batch_size,
                    nb_epoch=num_epochs, validation_split=0.1,
                    verbose=2, callbacks=[cback])
        if keras.__version__[0]=='1':
            history=model.fit_generator(
                    datagen.flow(x_train, y_train,batch_size=batch_size),
                    samples_per_epoch=x_train.shape[0], nb_epoch=num_epochs,
                    verbose=2, validation_split=0.1, callbacks=[cback])
    else:
        if keras.__version__[0]=='2':
            history=model.fit(x_train, y_train, batch_size=batch_size,validation_split=0.1, verbose=2, epochs=num_epochs,callbacks=[cback])
        if keras.__version__[0]=='1':
            history=model.fit(x_train, y_train,batch_size=batch_size,validation_split=0.1, verbose=2,nb_epoch=num_epochs,callbacks=[cback])

    dic={'hard': history.history}
    history_path = dir_path + "/history_output_model.pkl"
    with open(history_path,'wb') as handle:
        pickle.dump(dic, handle)

def evaluate_network(dir_path, dataset, x_test, y_test, batch_size, group_idx=None):
    weights_path= dir_path + "/output_model.h5"
    model = get_model(dataset, batch_size, group_idx=group_idx)
    model.load_weights(weights_path)
    opt = keras.optimizers.SGD()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    print("Test loss was %0.4f, test accuracy was %0.4f"%(score[0], score[1]))


dataset = "MNIST"
Train = True
Evaluate = True
batch_size = 100
num_epochs = 1000
x_train, y_train, x_test, y_test, use_generator = get_examples(dataset)

dir_path = "models/{}".format(dataset)
if Train:
    train_network(dir_path, dataset, x_train, y_train, num_epochs, use_generator, batch_size)
if Evaluate:
    evaluate_network(dir_path, dataset, x_test, y_test, batch_size)

