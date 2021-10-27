import math

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist


def compute_device_power(V: tf.constant, I_ind: tf.constant):
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


def compute_avg_crossbar_power(V: tf.constant, I_ind: tf.constant):
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
    P_avg = P_sum / tf.cast(tf.shape(V)[0], tf.float32)

    return P_avg


# learning rate schedule
def step_decay(epoch: int):
    initial_lrate = 0.025
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def normalize_img(image: tf.constant, label: str):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label
