import tensorflow as tf
from . import utils


def compute_I(V, G, V_ref, n_avg, n_std=tf.constant(0.0)):
    """Computes output currents of a crossbar consisting of devices suffering
    from I/V non-linearities.

    Parameters
    ----------
    V : ndarray
        Voltages of shape `p x m`.
    G : ndarray
        Conductances of shape `m x n`.
    V_ref :
        Reference voltage values of length r (in increasing order) or voltage
        at which the devices behave Ohmically.
    n_avg : tf.constant
        Average value of non-linearity parameter.
    n_std: tf.constant, optional
        Standard deviation of non-linearity parameter.

    Returns
    ----------
    I : ndarray
        Output currents of shape `p x n`.
    I_ind : ndarray
        Currents of shape `p x m x n` produced by each of the conductances in
        the crossbar array.
    """
    I_ind = compute_currents(n_avg, V_ref, G, V, n_std=n_std)
    I = add_I_BL(I_ind)

    return I, I_ind


def compute_currents(n_avg, V_ref, G, V, n_std=tf.constant(0.0)):
    """Compute current values by modelling I-V behaviour using nonlinearity
    parameter.

    Nonlinearity parameter n is defined as the current generated by a resistive
    device at voltage 2V divided by the current generated by the device at
    voltage V. We introduce voltage V_ref at which the generated amount of
    current equals the expected amount described by Ohm's law, i.e. I = VG.

    Parameters
    ----------
    n_avg : tf.constant
        Average value of non-linearity parameter.
    V_ref : float
        Voltage at which the devices behave Ohmically.
    G : ndarray
        Conductances of shape `m x n`.
    V : ndarray
        Voltages of shape `p x m`.
    n_std: tf.constant, optional
        Standard deviation of non-linearity parameter.

    Returns
    ----------
    I : ndarray
        Currents of shape `p x m x n` produced by each of the conductances in
        the crossbar array.
    """
    n = tf.random.normal(G.get_shape().as_list(), mean=n_avg, stddev=n_std, dtype=tf.float32)

    ohmic_current = V_ref * tf.expand_dims(G, axis=0)
    ratio = tf.expand_dims(V/V_ref, axis=-1)
    exponent = utils.tf_log2(n)

    I = ohmic_current * ratio ** exponent

    return I


def add_I_BL(I_ind):
    """Adds currents along the bit lines.

    Parameters
    ----------
    I_ind : ndarray
        Currents of shape `p x m x n` produced by each of the conductances in
        the crossbar array.

    Returns
    ----------
    I : ndarray
        Output currents of shape `p x n`.
    """
    I = tf.math.reduce_sum(I_ind, axis=1)
    return I

