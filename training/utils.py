import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
import scipy.io as sio
import math
import numpy as np


def compute_device_power(V, I_ind):
    """Computes power dissipated by individual devices in a crossbar.

    Parameters
    ----------
    V : tf.Tensor
        Voltages in shape (p x m) with p examples applied across m
        word lines.
    I_ind : tf.Tensor
        Currents in shape (p x m x n) generated by the individual
        devices in crossbar with m word lines and n bit lines.

    Returns
    ----------
    P_ind : tf.Tensor
        Power in shape (p x m x n).
    """
    # $P = VI$ for individual devices. All devices in the same word
    # line of the crossbar (row of G) are applied with the same voltage.
    P_ind = tf.einsum('ij,ijk->ijk', V, I_ind)

    return P_ind


def compute_avg_crossbar_power(V, I_ind):
    """Computes average power dissipated by a crossbar.

    Parameters
    ----------
    V : tf.Tensor
        Voltages in shape (p x m) with p examples applied across m
        word lines.
    I_ind : tf.Tensor
        Currents in shape (p x m x n) generated by the individual
        devices in crossbar with m word lines and n bit lines.

    Returns
    ----------
    P_avg : tf.Tensor
        Average power dissipated by a crossbar.
    """
    P = compute_device_power(V, I_ind)
    P_sum = tf.math.reduce_sum(P)
    # To get average power consumption **per crossbar** we divide by
    # number of examples.
    P_avg = P_sum/tf.cast(tf.shape(V)[0], tf.float32)

    return P_avg


def load_svhn(path_to_dataset):
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
