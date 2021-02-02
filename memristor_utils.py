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
import badmemristor
import badmemristor.nonideality

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
        G_eff = badmemristor.map.w_to_G_eff(weights, max_weight, G_min, G_max)
        G_eff_disturbed = badmemristor.nonideality.model.lognormal(G_eff,
                lognormal_G, lognormal_mean, lognormal_sigma, lognormal_rate,
                eff=True)
        disturbed_weights = badmemristor.map.G_eff_to_w(G_eff_disturbed,
                max_weight, G_max)
    else:
        G = badmemristor.map.w_to_G(weights, max_weight, G_min, G_max)
        G_disturbed = badmemristor.nonideality.model.lognormal(G, lognormal_G,
                lognormal_mean, lognormal_sigma, lognormal_rate)
        disturbed_weights = badmemristor.map.G_to_w(G_disturbed, max_weight, G_max)

    return disturbed_weights


def disturbance_faulty(weights, type_='unelectroformed', eff=True):
    max_weight = 2.5
    G_min = 1e-4
    G_max = 1e-3
    # An arbitrary proportion
    proportion = 0.05

    if eff:
        G_eff = badmemristor.map.w_to_G_eff(weights, max_weight, G_min, G_max)
        G_eff_disturbed = badmemristor.nonideality.D2D.faulty(G_eff,
                proportion, G_min=G_min, G_max=G_max, type_=type_, eff=eff)
        disturbed_weights = badmemristor.map.G_eff_to_w(G_eff_disturbed,
                max_weight, G_max)
    else:
        G = badmemristor.map.w_to_G(weights, max_weight, G_min, G_max)
        G_disturbed = badmemristor.nonideality.D2D.faulty(G, proportion,
                G_min=G_min, G_max=G_max, type_=type_, eff=eff)
        disturbed_weights = badmemristor.map.G_to_w(G_disturbed, max_weight, G_max)

    return disturbed_weights


def disturbed_outputs_i_v_non_linear(x, weights, eff=True):
    # TODO: test if works correctly.

    # Reference values for conductances (in siemens) of devices in certain states.
    # These refer to conductances in the linear operating region (usually, at low
    # voltages). If there will be unelectroformed devices, a value for 0 siemens
    # should also be included.
    G_ref = np.array([0, 1e-4, 3e-4, 5e-4])
    # Voltage values (in volts) at which the currents were measured.
    V_ref = np.array([-1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2])
    # Currents (in amps). Row corresponds to reference conductance states and
    # columns to reference voltages. For example, a device with reference
    # conductance of 3e-4 S would produce 50e-5 A of current if a voltage of 1.2 V
    # was applied across it.
    I_ref = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-15e-5, -11e-5, -7e-5, -3e-5, 0, 3e-5, 7e-5, 11e-5, 15e-5],
        [-50e-5, -35e-5, -22e-5, -9e-5, 0, 9e-5, 22e-5, 35e-5, 50e-5],
        [-90e-5, -60e-5, -37e-5, -15e-5, 0, 15e-5, 37e-5, 60e-5, 90e-5]
        ])

    # I picked this myself, but you may want to create a function that decides
    # how large `max_weight` should be, given an array of weights.
    max_weight = np.max(np.abs(weights))
    # These are arbitrary, but you may be given some values to work with or may need
    # to research what these values could be.
    G_min = 1e-4
    G_max = 5e-4
    k_V = 1.05

    # Mapping weights onto conductances.
    if eff:
        G = badmemristor.map.w_to_G_eff(weights, max_weight, G_min, G_max)
    else:
        G = badmemristor.map.w_to_G(weights, max_weight, G_min, G_max)

    # Mapping inputs onto voltages.
    V = badmemristor.map.x_to_V(x, k_V)

    # Computing currents
    I = badmemristor.nonideality.i_v_non_linear.compute_I(
            V, G, V_ref, G_ref, I_ref, eff=eff)

    # Converting to outputs.
    if eff:
        y_disturbed = badmemristor.map.I_total_to_y(I, k_V, max_weight, G_max)
    else:
        y_disturbed = badmemristor.map.I_to_y(I, k_V, max_weight, G_max)

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
        # Non-ideality-aware training
        bias = tf.expand_dims(self.b, axis=0)
        combined_weights = tf.concat([self.w, bias], 0)
        ones = tf.ones([tf.shape(x)[0], 1])
        inputs = tf.concat([x, ones], 1)
        # self.out = K.dot(inputs, self.apply_weight_disturbance(combined_weights))
        self.out = self.apply_output_disturbance(inputs, combined_weights)

        #self.out = K.dot(x, self.w) + self.b # Vanilla CNN
        return self.out

    @tf_custom_gradient_method
    def apply_weight_disturbance(self, undisturbed_w):
        # Forward propagation

        disturbed_w = tf.py_function(func=disturbance, inp=[undisturbed_w], Tout=tf.float32)
        disturbed_w.set_shape((self.n_in+1, self.n_out)) # Outputs to py_function do not have shape defined

        def custom_grad(disturbed_w_grad):
            # Backward propagation
            undisturbed_w_grad = disturbed_w_grad # Does nothing
            return undisturbed_w_grad
        return disturbed_w, custom_grad

    @tf_custom_gradient_method
    def apply_output_disturbance(self, inputs, weights):
        # Forward propagation
        disturbed_outputs = tf.py_function(func=disturbed_outputs_i_v_non_linear, inp=[inputs, weights], Tout=tf.float32)
        disturbed_outputs.set_shape((inputs.shape.as_list()[0], weights.shape.as_list()[1])) # Outputs to py_function do not have shape defined

        def custom_grad(disturbed_outputs_grad):
            # Backward propagation
            # TODO (Erwei)
            inputs_grad = tf.matmul(disturbed_outputs_grad, tf.transpose(weights))
            weights_grad = tf.matmul(tf.transpose(inputs), disturbed_outputs_grad)

            return (inputs_grad, weights_grad)

        return disturbed_outputs, custom_grad

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
