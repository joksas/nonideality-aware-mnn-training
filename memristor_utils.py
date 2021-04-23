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
    V_ref = tf.constant(0.0265)
    G_min = tf.constant(1/191000)
    G_max = tf.constant(1/139000)
    n_param = tf.constant(7.4)
    k_V = 2*V_ref

    G = badmemristor_tf.map.w_to_G(weights, max_weight, G_min, G_max, scheme="differential")

    # Mapping inputs onto voltages.
    V = badmemristor_tf.map.x_to_V(x, k_V)

    # Computing currents
    I = badmemristor_tf.nonideality.i_v_non_linear.compute_I(
            V, G, V_ref, G_min, G_max, n_param=n_param, eff=False, model="nonlinear_param")

    # Converting to outputs.
    y_disturbed = badmemristor_tf.map.I_to_y(I, k_V, max_weight, G_max, G_min, scheme="differential")

    tf.debugging.assert_all_finite(
        x, "nan in outputs", name=None
    )

    return y_disturbed


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

        self.w = self.add_weight(
            shape=(self.n_in,self.n_out),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv),
            name="weights",
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.n_out,),
            initializer=tf.keras.initializers.Constant(value=0.0),
            name="biasess",
            trainable=True,
        )


    def call(self, x,mask=None):

        # Clip inputs within 0 and 1
        x = tf.clip_by_value(x, 0.0, 1.0)

        # Non-ideality-aware training
        bias = tf.expand_dims(self.b, axis=0)
        combined_weights = tf.concat([self.w, bias], 0)
        ones = tf.ones([tf.shape(x)[0], 1])
        inputs = tf.concat([x, ones], 1)

        self.out = self.apply_output_disturbance(inputs, combined_weights)

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

