import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Layer
from tensorflow.keras import backend as K
import numpy as np
import crossbar
from . import utils


def get_model(iterator):
	if iterator.dataset=='MNIST':
		model=Sequential()
		model.add(MemristorDense(784, 25, iterator, input_shape=[784]))
        # We will try to introduce non-linearities using dense layers.
		model.add(Activation('sigmoid'))
		model.add(MemristorDense(int(model.output.get_shape()[1]), 10, iterator))
		model.add(Activation('softmax'))
	else:
		raise("Dataset {} is not recognised!".format(iterator.dataset))
	return model


class MemristorDense(Layer):
    def __init__(self, n_in, n_out, iterator, **kwargs):
        self.n_in=n_in
        self.n_out=n_out
        self.iterator = iterator
        super(MemristorDense, self).__init__(**kwargs)

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
        if self.iterator.training.is_regularized:
            reg_gamma = 1e-4
            kwargs["regularizer"] = tf.keras.regularizers.l1(reg_gamma)

        if self.iterator.training.is_nonideal():
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
                name="biases_pos",
                trainable=True,
                **kwargs
            )

            self.b_neg = self.add_weight(
                shape=(self.n_out,),
                initializer=tf.keras.initializers.Constant(value=0.5),
                name="biases_neg",
                trainable=True,
                **kwargs
            )
        else:
            self.w = self.add_weight(
                shape=(self.n_in,self.n_out),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv),
                name="weights",
                trainable=True,
                **kwargs
                )

            self.b = self.add_weight(
                shape=(self.n_out,),
                initializer=tf.keras.initializers.Constant(value=0.0),
                name="biases",
                trainable=True,
                **kwargs
            )

    def call(self, x, mask=None):
        ones = tf.ones([tf.shape(x)[0], 1])
        inputs = tf.concat([x, ones], 1)

        if self.iterator.training.is_nonideal():
            b_pos = tf.expand_dims(self.b_pos, axis=0)
            b_neg = tf.expand_dims(self.b_neg, axis=0)
            combined_weights_pos = tf.concat([self.w_pos, b_pos], 0)
            combined_weights_neg = tf.concat([self.w_neg, b_neg], 0)

            # Interleave positive and negative weights
            combined_weights = tf.reshape(
                    tf.concat([
                        combined_weights_pos[..., tf.newaxis],
                        combined_weights_neg[..., tf.newaxis]
                        ], axis=-1),
                    [tf.shape(combined_weights_pos)[0], -1]
                    )
        else:
            bias = tf.expand_dims(self.b, axis=0)
            combined_weights = tf.concat([self.w, bias], 0)

        self.out = self.memristive_outputs(inputs, combined_weights)

        return self.out

    def memristive_outputs(self, x, weights):
        # Mapping inputs onto voltages.
        V_ref = tf.constant(0.25)
        k_V = 2*V_ref
        V = crossbar.map.x_to_V(x, k_V)

        # Mapping weights onto conductances.
        max_weight = tf.math.reduce_max(tf.math.abs(weights))
        G_min = tf.constant(self.iterator.G_min)
        G_max = tf.constant(self.iterator.G_max)
        if self.iterator.training.is_nonideal():
            G = crossbar.map.w_params_to_G(weights, max_weight, G_min, G_max)
        else:
            G = crossbar.map.w_to_G(weights, max_weight, G_min, G_max)

        if self.iterator.current_stage().iv_nonlinearity is not None:
            n_avg = tf.constant(self.iterator.training.iv_nonlinearity.n_avg)
            n_std = tf.constant(self.iterator.training.iv_nonlinearity.n_std)
            # Computing currents
            I, I_ind = crossbar.nonlinear_IV.compute_I(
                    V, G, V_ref, G_min, G_max, n_avg, n_std=n_std)

            if not self.iterator.is_training:
                power_path = self.iterator.power_path()
                open(power_path, "a").close()
                P_avg = utils.compute_avg_crossbar_power(V, I_ind)
                tf.print(P_avg, output_stream="file://{}".format(power_path))
        else:
            # Ideal case for computing output currents.
            # TODO: implement function for computing currents in every branch.
            I = K.dot(V, G)

        # Converting to outputs.
        y_disturbed = crossbar.map.I_to_y(I, k_V, max_weight, G_max, G_min)

        tf.debugging.assert_all_finite(
            y_disturbed, "nan in outputs", name=None
        )

        return y_disturbed


    def get_output_shape_for(self,input_shape):
        return (input_shape[0], self.n_out)
    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.n_out)

