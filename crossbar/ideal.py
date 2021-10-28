import tensorflow as tf

from . import utils


def compute_I_all(V: tf.Tensor, G: tf.Tensor):
    """Computes output and device currents of an ideal crossbar.

    Parameters
    ----------
    V : ndarray
        Voltages of shape `p x m`.
    G : ndarray
        Conductances of shape `m x n`.

    Returns
    ----------
    I : ndarray
        Output currents of shape `p x n`.
    I_ind : ndarray
        Currents of shape `p x m x n` produced by each of the conductances in
        the crossbar array.
    """
    I_ind = tf.expand_dims(V, axis=-1) * tf.expand_dims(G, axis=0)
    I = utils.add_I_BL(I_ind)
    return I, I_ind


def compute_I(V: tf.Tensor, G: tf.Tensor):
    """Computes output currents of an ideal crossbar.

    Parameters
    ----------
    V : ndarray
        Voltages of shape `p x m`.
    G : ndarray
        Conductances of shape `m x n`.

    Returns
    ----------
    ndarray
        Output currents of shape `p x n`.
    """
    return tf.tensordot(V, G, axes=1)
