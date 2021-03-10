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
import badmemristor_tf.nonideality

# A decorator for customising gradients
def tf_custom_gradient_method(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, '_tf_custom_gradient_wrappers'):
            self._tf_custom_gradient_wrappers = {}
        if f not in self._tf_custom_gradient_wrappers:
            self._tf_custom_gradient_wrappers[f] = tf.custom_gradient(lambda *a, **kw: f(self, *a, **kw))
        return self._tf_custom_gradient_wrappers[f](*args, **kwargs)
    return wrapped


def disturbance_lognormal(weights, eff=True):
    # I picked this myself, but you may want to create a function that decides
    # how large `max_weight` should be, given an array of weights.
    # TODO: It would make more sense to pick it separately for each weight
    # array. E.g. set it to the highest-absolute-value weight.
    max_weight = 2.5
    # These are arbitrary, but you may be given some values to work with or may
    # need to research what these values could be.
    G_min = 1e-4
    G_max = 1e-3
    # Some arbitrary values for lognormal disturbance (again, you might be
    # given some or might need to find them yourself).
    lognormal_G = [1e-4, 5e-4, 9e-4]
    lognormal_mean = [-5, -4, -3]
    lognormal_sigma = [0.5, 0.3, 0.1]
    lognormal_rate = [0.5, 0.6, 0.5]

    if eff:
        G_eff = badmemristor_tf.map.w_to_G_eff(weights, max_weight, G_min, G_max)
        G_eff_disturbed = badmemristor_tf.nonideality.model.lognormal(G_eff,
                lognormal_G, lognormal_mean, lognormal_sigma, lognormal_rate,
                eff=True)
        disturbed_weights = badmemristor_tf.map.G_eff_to_w(G_eff_disturbed,
                max_weight, G_max)
    else:
        G = badmemristor_tf.map.w_to_G(weights, max_weight, G_min, G_max)
        G_disturbed = badmemristor_tf.nonideality.model.lognormal(G, lognormal_G,
                lognormal_mean, lognormal_sigma, lognormal_rate)
        disturbed_weights = badmemristor_tf.map.G_to_w(G_disturbed, max_weight, G_max)

    return disturbed_weights


def disturbance_faulty(weights, type_='unelectroformed', eff=True):
    max_weight = 2.5
    G_min = 1e-4
    G_max = 1e-3
    # An arbitrary proportion
    proportion = 0.05

    if eff:
        G_eff = badmemristor_tf.map.w_to_G_eff(weights, max_weight, G_min, G_max)
        G_eff_disturbed = badmemristor_tf.nonideality.D2D.faulty(G_eff,
                proportion, G_min=G_min, G_max=G_max, type_=type_, eff=eff)
        disturbed_weights = badmemristor_tf.map.G_eff_to_w(G_eff_disturbed,
                max_weight, G_max)
    else:
        G = badmemristor_tf.map.w_to_G(weights, max_weight, G_min, G_max)
        G_disturbed = badmemristor_tf.nonideality.D2D.faulty(G, proportion,
                G_min=G_min, G_max=G_max, type_=type_, eff=eff)
        disturbed_weights = badmemristor_tf.map.G_to_w(G_disturbed, max_weight, G_max)

    return disturbed_weights


def disturbed_outputs_i_v_non_linear(x, weights):
    max_weight = tf.math.reduce_max(tf.math.abs(weights))
    V_ref = tf.constant(0.25)

    is_low_resistance = True

    if is_low_resistance:
        # Low resistance SiO_x.
        G_min = tf.constant(0.1)
        G_max = tf.constant(1.0)
        n_avg = tf.constant(2.18)
        n_std = tf.constant(0.115)
    else:
        # High resistance SiO_x.
        G_min = tf.constant(1/1430000)
        G_max = tf.constant(1/368000)
        n_avg = tf.constant(2.88)
        n_std = tf.constant(0.363)

    eff = True
    # Mapping weights onto conductances.
    if eff:
        tf.print("W =", weights)
        G = badmemristor_tf.map.w_to_G_eff(weights, max_weight, G_min, G_max, scheme="differential")
        # Apply G_min
        G = tf.math.abs(G) + G_min
        tf.print("G =", G)
        A = badmemristor_tf.map.w_to_G(weights, max_weight, G_min, G_max, scheme="differential")
        tf.print("A =", A)

    k_V = 2*V_ref

    # Mapping inputs onto voltages.
    V = badmemristor_tf.map.x_to_V(x, k_V)

    # Computing currents
    I, _ = badmemristor_tf.nonideality.i_v_non_linear.compute_I(
            V, G, V_ref, G_min, G_max, n_avg=n_avg, n_std=n_std, eff=False, model="nonlinear_param")

    # Converting to outputs.
    y_disturbed = badmemristor_tf.map.I_to_y(I, k_V, max_weight, G_max, G_min, scheme="differential")

    tf.debugging.assert_all_finite(
        x, "nan in outputs", name=None
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


def disturbance(weights, type_='lognormal', faulty_type='unelectroformed',
        eff=True):
    if type_ == 'lognormal':
        disturbed_weights = disturbance_lognormal(weights, eff=eff)
    elif type == 'faulty':
        disturbed_weights = disturbance_faulty(weights, type_=faulty_type,
                eff=eff)
    else:
        raise ValueError(
                'Disturbance type "{}" is not supported!'.format(type_)
                )

    return disturbed_weights


class memristor_dense(Layer):
    def __init__(self,n_in,n_out,**kwargs):
        self.n_in=n_in
        self.n_out=n_out
        super(memristor_dense,self).__init__(**kwargs)

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

        self.w_pos = self.add_weight(
            shape=(self.n_in,self.n_out),
            initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=stdv),
            #initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=None),
            name="weights_pos",
            trainable=True,
        )

        self.w_neg = self.add_weight(
            shape=(self.n_in,self.n_out),
            initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=stdv),
            #initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=None),
            name="weights_neg",
            trainable=True,
        )

        self.b_pos = self.add_weight(
            shape=(self.n_out,),
            initializer=tf.keras.initializers.Constant(value=0.5),
            name="biasess_pos",
            trainable=True,
        )

        self.b_neg = self.add_weight(
            shape=(self.n_out,),
            initializer=tf.keras.initializers.Constant(value=0.5),
            name="biasess_neg",
            trainable=True,
        )


    def call(self, x,mask=None):

        # Clip inputs within 0 and 1
        x = tf.clip_by_value(x, 0.0, 1.0)

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
        disturbed_outputs = disturbed_outputs_i_v_non_linear(inputs, weights)
        return disturbed_outputs

    def get_output_shape_for(self,input_shape):
        return (input_shape[0], self.n_out)
    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.n_out)

class my_flat(Layer):
    def __init__(self,**kwargs):
        super(my_flat,self).__init__(**kwargs)
    def build(self, input_shape):
        return

    def call(self, x, mask=None):
        self.out=tf.reshape(x,[-1,np.prod(x.get_shape().as_list()[1:])])
        return self.out
    def  compute_output_shape(self,input_shape):
        shpe=(input_shape[0],int(np.prod(input_shape[1:])))
        return shpe

