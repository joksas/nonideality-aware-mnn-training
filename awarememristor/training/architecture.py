import numpy as np
import tensorflow as tf
from tensorflow.keras import constraints, layers, models

from awarememristor import crossbar
from awarememristor.training import utils


def get_model(iterator, custom_weights=None, custom_weights_path=None):
    num_hidden_neurons = 25
    if iterator.dataset == "mnist":
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=(28, 28)))
        model.add(MemristorDense(784, num_hidden_neurons, iterator))
        model.add(layers.Activation("sigmoid"))
        model.add(MemristorDense(num_hidden_neurons, 10, iterator))
        model.add(layers.Activation("softmax"))
    elif iterator.dataset == "cifar10":
        model = models.Sequential()

        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        # Ensure inputs bounded between 0 and 1 for first memristive layer.
        model.add(layers.Conv2D(64, (3, 3), activation=tf.keras.layers.ReLU(max_value=1.0)))

        # Fully connected layers
        model.add(layers.Flatten())
        model.add(MemristorDense(1024, num_hidden_neurons, iterator))
        model.add(layers.Activation("sigmoid"))
        model.add(MemristorDense(num_hidden_neurons, 10, iterator))
        model.add(layers.Activation("softmax"))
    else:
        raise ValueError(f"Dataset {iterator.dataset} is not recognised!")

    if custom_weights is not None:
        model.set_weights(custom_weights)
    elif custom_weights_path is not None:
        model.load_weights(custom_weights_path)
    elif not iterator.is_training:
        model.load_weights(iterator.weights_path())

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


class MemristorDense(layers.Layer):
    def __init__(self, n_in, n_out, iterator, **kwargs):
        self.n_in = n_in
        self.n_out = n_out
        self.iterator = iterator
        super(MemristorDense, self).__init__(**kwargs)

    # Adding this function removes an issue with custom layer checkpoint
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_in": self.n_in,
                "n_out": self.n_out,
            }
        )
        return config

    # Create trainable weights and biases
    def build(self, input_shape):
        stdv = 1 / np.sqrt(self.n_in)

        kwargs = {}
        if self.iterator.training.is_regularized:
            reg_gamma = 1e-4
            kwargs["regularizer"] = tf.keras.regularizers.l1(reg_gamma)

        if self.iterator.training.uses_double_weights():
            self.w_pos = self.add_weight(
                shape=(self.n_in, self.n_out),
                initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=stdv),
                name="weights_pos",
                trainable=True,
                constraint=constraints.NonNeg(),
                **kwargs,
            )

            self.w_neg = self.add_weight(
                shape=(self.n_in, self.n_out),
                initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=stdv),
                name="weights_neg",
                trainable=True,
                constraint=constraints.NonNeg(),
                **kwargs,
            )

            self.b_pos = self.add_weight(
                shape=(self.n_out,),
                initializer=tf.keras.initializers.Constant(value=0.5),
                name="biases_pos",
                trainable=True,
                constraint=constraints.NonNeg(),
                **kwargs,
            )

            self.b_neg = self.add_weight(
                shape=(self.n_out,),
                initializer=tf.keras.initializers.Constant(value=0.5),
                name="biases_neg",
                trainable=True,
                constraint=constraints.NonNeg(),
                **kwargs,
            )
        else:
            self.w = self.add_weight(
                shape=(self.n_in, self.n_out),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv),
                name="weights",
                trainable=True,
                **kwargs,
            )

            self.b = self.add_weight(
                shape=(self.n_out,),
                initializer=tf.keras.initializers.Constant(value=0.0),
                name="biases",
                trainable=True,
                **kwargs,
            )

    def combined_weights(self):
        if self.iterator.training.uses_double_weights():
            b_pos = tf.expand_dims(self.b_pos, axis=0)
            b_neg = tf.expand_dims(self.b_neg, axis=0)
            combined_weights_pos = tf.concat([self.w_pos, b_pos], 0)
            combined_weights_neg = tf.concat([self.w_neg, b_neg], 0)

            # Interleave positive and negative weights
            combined_weights = tf.reshape(
                tf.concat(
                    [
                        combined_weights_pos[..., tf.newaxis],
                        combined_weights_neg[..., tf.newaxis],
                    ],
                    axis=-1,
                ),
                [tf.shape(combined_weights_pos)[0], -1],
            )
        else:
            bias = tf.expand_dims(self.b, axis=0)
            combined_weights = tf.concat([self.w, bias], 0)

        return combined_weights

    def call(self, x, mask=None):
        if (
            not self.iterator.training.uses_double_weights()
            and not self.iterator.current_stage().is_nonideal()
        ):
            return tf.tensordot(x, self.w, axes=1) + self.b

        ones = tf.ones([tf.shape(x)[0], 1])
        inputs = tf.concat([x, ones], 1)

        self.out = self.memristive_outputs(inputs, self.combined_weights())

        return self.out

    def memristive_outputs(self, x, weights):
        current_stage = self.iterator.current_stage()

        if current_stage.is_nonideal():
            G_off = current_stage.G_off
            G_on = current_stage.G_on
            k_V = current_stage.k_V()
        else:
            # Handle case when training is aware, but inference assumes no
            # nonidealities. This will not affect accuracy, but will affect
            # power consumption.
            G_off = self.iterator.training.G_off
            G_on = self.iterator.training.G_on
            k_V = self.iterator.training.k_V()

        # Mapping inputs onto voltages.
        V = crossbar.map.x_to_V(x, k_V)

        # Mapping weights onto conductances.
        if self.iterator.training.uses_double_weights():
            G, max_weight = crossbar.map.double_w_to_G(weights, G_off, G_on)
        else:
            G, max_weight = crossbar.map.w_to_G(weights, G_off, G_on, current_stage.mapping_rule)

        # Linearity-preserving nonidealities
        for nonideality in current_stage.nonidealities:
            if isinstance(nonideality, crossbar.nonidealities.LinearityPreserving):
                G = nonideality.disturb_G(G)

        # Linearity-nonpreserving nonidealities
        I = None
        I_ind = None
        for nonideality in current_stage.nonidealities:
            if isinstance(nonideality, crossbar.nonidealities.LinearityNonpreserving):
                I, I_ind = nonideality.compute_I(V, G)

        # Ideal case for computing output currents.
        if I is None or I_ind is None:
            if self.iterator.is_training:
                I = crossbar.ideal.compute_I(V, G)
            else:
                I, I_ind = crossbar.ideal.compute_I_all(V, G)

        if not self.iterator.is_training and not self.iterator.is_callback:
            power_path = self.iterator.power_path()
            P_avg = utils.compute_avg_crossbar_power(V, I_ind)
            with open(power_path, mode="a", encoding="utf-8"):
                tf.print(P_avg, output_stream=f"file://{power_path}")

        # Converting to outputs.
        y_disturbed = crossbar.map.I_to_y(I, k_V, max_weight, G_on, G_off)

        return y_disturbed

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n_out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_out)
