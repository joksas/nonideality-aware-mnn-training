import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import functools

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D,Cropping2D
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.framework import ops

# Import memristor non-idealities
import badmemristor_tf


def disturbed_outputs_i_v_non_linear(x, weights, group_idx=None, log_dir_full_path=None):
    if group_idx is None:
        group_idx = 0

    max_weight = tf.math.reduce_max(tf.math.abs(weights))
    V_ref = tf.constant(0.25)

    G_min_lst = tf.constant([1/983.3, 1/10170, 1/1401000])
    G_max_lst = tf.constant([1/281.3, 1/2826, 1/385700])
    n_avg_lst = tf.constant([2.132, 2.596, 2.986])
    n_std_lst = tf.constant([0.095, 0.088, 0.378])

    G_min = G_min_lst[group_idx]
    G_max = G_max_lst[group_idx]
    n_avg= n_avg_lst[group_idx]
    n_std= n_std_lst[group_idx]

    # Mapping weights onto conductances.
    G = badmemristor_tf.map.w_params_to_G(weights, max_weight, G_min, G_max)

    k_V = 2*V_ref

    # Mapping inputs onto voltages.
    V = badmemristor_tf.map.x_to_V(x, k_V)

    # Computing currents
    I, I_ind = badmemristor_tf.nonlinear_IV.compute_I(
            V, G, V_ref, G_min, G_max, n_avg, n_std=n_std)
    if log_dir_full_path is not None:
        log_file_full_path = "{}/power.csv".format(log_dir_full_path)
        open(log_file_full_path, "a").close()
        P_avg = compute_avg_crossbar_power(V, I_ind)
        tf.print(P_avg, output_stream="file://{}".format(log_file_full_path))

    # Converting to outputs.
    y_disturbed = badmemristor_tf.map.I_to_y(I, k_V, max_weight, G_max, G_min)

    tf.debugging.assert_all_finite(
        y_disturbed, "nan in outputs", name=None
    )

    return y_disturbed


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


class memristor_dense(Layer):
    def __init__(self, n_in, n_out, group_idx=None, is_regularized=True, log_dir_full_path=None, **kwargs):
        self.n_in=n_in
        self.n_out=n_out
        self.group_idx = group_idx
        self.is_regularized = is_regularized
        self.log_dir_full_path = log_dir_full_path
        super(memristor_dense, self).__init__(**kwargs)

    # Adding this funcion removes an issue with custom layer checkpoint
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_in': self.n_in,
            'n_out': self.n_out,
        })
        return config

    # Create trainable weights and biases
    def build(self, input_shape):
        stdv=1/np.sqrt(self.n_in)
        kwargs = {}
        if self.is_regularized:
            reg_gamma = 1e-4
            kwargs["regularizer"] = tf.keras.regularizers.l1(reg_gamma)

        self.w_pos = self.add_weight(
            shape=(self.n_in,self.n_out),
            initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=stdv),
            name="weights_pos",
            trainable=True,
            **kwargs
        )

        self.w_neg = self.add_weight(
            shape=(self.n_in,self.n_out),
            initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=stdv),
            name="weights_neg",
            trainable=True,
            **kwargs
        )

        self.b_pos = self.add_weight(
            shape=(self.n_out,),
            initializer=tf.keras.initializers.Constant(value=0.5),
            name="biasess_pos",
            trainable=True,
            **kwargs
        )

        self.b_neg = self.add_weight(
            shape=(self.n_out,),
            initializer=tf.keras.initializers.Constant(value=0.5),
            name="biasess_neg",
            trainable=True,
            **kwargs
        )


    def call(self, x,mask=None):

        # Clip inputs within 0 and 1
        #x = tf.clip_by_value(x, 0.0, 1.0)

        # Non-ideality-aware training
        bias_pos = tf.expand_dims(self.b_pos, axis=0)
        bias_neg = tf.expand_dims(self.b_neg, axis=0)
        combined_weights_pos = tf.concat([self.w_pos, bias_pos], 0)
        combined_weights_neg = tf.concat([self.w_neg, bias_neg], 0)
        ones = tf.ones([tf.shape(x)[0], 1])
        inputs = tf.concat([x, ones], 1)

        is_aware = True
        if is_aware:
            # Interleave positive and negative weights
            combined_weights = tf.reshape(
            tf.concat([combined_weights_pos[...,tf.newaxis], combined_weights_neg[...,tf.newaxis]], axis=-1),
            [tf.shape(combined_weights_pos)[0],-1])

            self.out = self.apply_output_disturbance(inputs, combined_weights)
        else:
            self.out = K.dot(x, self.w) + self.b

        return self.out

    def apply_output_disturbance(self, inputs, weights):
        disturbed_outputs = disturbed_outputs_i_v_non_linear(inputs, weights, group_idx=self.group_idx, log_dir_full_path=self.log_dir_full_path)
        return disturbed_outputs

    def get_output_shape_for(self,input_shape):
        return (input_shape[0], self.n_out)
    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.n_out)

