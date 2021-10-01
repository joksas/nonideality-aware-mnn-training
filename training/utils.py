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
    P_ind = tf.einsum("ij,ijk->ijk", V, I_ind)

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


def get_examples(dataset):
    if dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
        num_classes = 10
        use_generator = False
    elif dataset == "CIFAR-10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        use_generator = True
        num_classes = 10
    else:
        raise ValueError("Dataset should be one of the following: [MNIST, CIFAR-10].")

    x_train, x_test = x_train/255.0, x_test/255.0
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    return x_train, y_train, x_test, y_test, use_generator


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.025
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
