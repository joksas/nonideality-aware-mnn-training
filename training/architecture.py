import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import crossbar
from . import utils


def get_model(iterator):
    num_hidden_neurons = 25
    if iterator.dataset == "MNIST":
        model = models.Sequential()
        model.add(MemristorDense(784, num_hidden_neurons, iterator, input_shape=[784]))
        model.add(layers.Activation("sigmoid"))
        model.add(MemristorDense(num_hidden_neurons, 10, iterator))
        model.add(layers.Activation("softmax"))

        opt = tf.keras.optimizers.SGD()
    elif iterator.dataset == "CIFAR-10":
        model = models.Sequential()

        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="sigmoid"))

        # Fully connected layers
        model.add(layers.Flatten())
        model.add(MemristorDense(1024, num_hidden_neurons, iterator))
        model.add(layers.Activation("sigmoid"))
        model.add(MemristorDense(num_hidden_neurons, 10, iterator))
        model.add(layers.Activation("softmax"))

        opt = tf.keras.optimizers.Adam()
    else:
        raise "Dataset {} is not recognised!".format(iterator.dataset)

    if not iterator.is_training:
        model.load_weights(iterator.weights_path())

    model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
    return model


class MemristorDense(layers.Layer):
    def __init__(self, n_in, n_out, iterator, **kwargs):
        self.n_in=n_in
        self.n_out=n_out
        self.iterator = iterator
        super(MemristorDense, self).__init__(**kwargs)

    # Adding this function removes an issue with custom layer checkpoint
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "n_in": self.n_in,
            "n_out": self.n_out,
            })
        return config

    # Create trainable weights and biases
    def build(self, input_shape):
        stdv=1/np.sqrt(self.n_in)

        kwargs = {}
        if self.iterator.training.is_regularized:
            reg_gamma = 1e-4
            kwargs["regularizer"] = tf.keras.regularizers.l1(reg_gamma)

        if self.iterator.training.is_aware():
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
        # tf.print("x shape:", x.shape)
        # tf.print("x:", x)
        ones = tf.ones([tf.shape(x)[0], 1])
        inputs = tf.concat([x, ones], 1)

        if self.iterator.training.is_aware():
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
        G_min = tf.constant(self.iterator.G_min)
        G_max = tf.constant(self.iterator.G_max)
        if self.iterator.training.is_aware():
            G, max_weight = crossbar.map.w_params_to_G(weights, G_min, G_max)
        else:
            G, max_weight = crossbar.map.w_to_G(weights, G_min, G_max)

        current_stage = self.iterator.current_stage()

        # Linearity-preserving nonidealities
        if current_stage.stuck_at_G_min is not None or current_stage.stuck_at_G_max is not None:
            if current_stage.stuck_at_G_min is not None:
                G = crossbar.faulty_devices.random_devices_stuck(
                        G, G_min, current_stage.stuck_at_G_min.p)
            elif current_stage.stuck_at_G_max is not None:
                G = crossbar.faulty_devices.random_devices_stuck(
                        G, G_max, current_stage.stuck_at_G_max.p)

        # Other nonidealities
        if current_stage.iv_nonlinearity is not None:
            n_avg = tf.constant(current_stage.iv_nonlinearity.n_avg)
            n_std = tf.constant(current_stage.iv_nonlinearity.n_std)
            # Computing currents
            I, I_ind = crossbar.nonlinear_IV.compute_I_all(
                    V, G, V_ref, n_avg, n_std=n_std)
        else:
            # Ideal case for computing output currents.
            if self.iterator.is_training:
                I = crossbar.ideal.compute_I(V, G)
            else:
                I, I_ind = crossbar.ideal.compute_I_all(V, G)

        if not self.iterator.is_training:
            power_path = self.iterator.power_path()
            open(power_path, "a").close()
            P_avg = utils.compute_avg_crossbar_power(V, I_ind)
            tf.print(P_avg, output_stream="file://{}".format(power_path))

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
