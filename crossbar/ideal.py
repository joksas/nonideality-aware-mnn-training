import tensorflow as tf

from . import utils


def compute_I_all(V: tf.Tensor, G: tf.Tensor) -> tf.Tensor:
    """Compute output and device currents of an ideal crossbar.

    Args:
        V: Voltages of shape `p x m`.
        G: Conductances of shape `m x n`.

    Returns:
        I: Output currents of shape `p x n`.
        I_ind: Currents of shape `p x m x n` produced by each of the conductances in the crossbar
            array.
    """
    I_ind = tf.expand_dims(V, axis=-1) * tf.expand_dims(G, axis=0)
    I = utils.add_I_BL(I_ind)
    return I, I_ind


def compute_I(V: tf.Tensor, G: tf.Tensor) -> tf.Tensor:
    """Compute output currents of an ideal crossbar.

    Args:
        V: Voltages of shape `p x m`.
        G: Conductances of shape `m x n`.

    Returns:
        Output currents of shape `p x n`.
    """
    return tf.tensordot(V, G, axes=1)
