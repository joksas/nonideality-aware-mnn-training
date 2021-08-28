import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Layer
import numpy as np
import crossbar


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
        # disturbed_outputs = disturbed_outputs_i_v_non_linear(inputs, weights, group_idx=self.group_idx, log_dir_full_path=self.log_dir_full_path)
        # return disturbed_outputs
        return self.nonlinear_iv_outputs(inputs, weights)

    def nonlinear_iv_outputs(self, x, weights):
        max_weight = tf.math.reduce_max(tf.math.abs(weights))
        V_ref = tf.constant(0.25)

        G_min = tf.constant(self.iterator.G_min)
        G_max = tf.constant(self.iterator.G_max)
        n_avg = tf.constant(self.iterator.training.iv_nonlinearity.n_avg)
        n_std = tf.constant(self.iterator.training.iv_nonlinearity.n_std)

        # Mapping weights onto conductances.
        G = crossbar.map.w_params_to_G(weights, max_weight, G_min, G_max)

        k_V = 2*V_ref

        # Mapping inputs onto voltages.
        V = crossbar.map.x_to_V(x, k_V)

        # Computing currents
        I, I_ind = crossbar.nonlinear_IV.compute_I(
                V, G, V_ref, G_min, G_max, n_avg, n_std=n_std)
        # if True:
        #     log_file_full_path = "{}/power.csv".format(log_dir_full_path)
        #     open(log_file_full_path, "a").close()
        #     P_avg = compute_avg_crossbar_power(V, I_ind)
        #     tf.print(P_avg, output_stream="file://{}".format(log_file_full_path))

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

