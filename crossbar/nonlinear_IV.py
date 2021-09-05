import tensorflow as tf


def compute_I(V, G, V_ref, G_min, G_max, n_avg, n_std=tf.constant(0.0)):
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
    G_min : float
        Minimum conductance of electroformed memristors.
    G_max : float
        Maximum conductance of electroformed memristors.
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
    I_ind = compute_currents(G_min, G_max, n_avg, V_ref, G, V, n_std=n_std)
    I = add_I_BL(I_ind)

    return I, I_ind


def compute_currents(G_min, G_max, n_avg, V_ref, G, V, n_std=tf.constant(0.0)):
    """Compute current values by modelling I-V behaviour using nonlinearity
    parameter.

    Parameters
    ----------
    G_min : float
        Minimum conductance of electroformed memristors.
    G_max : float
        Maximum conductance of electroformed memristors.
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
    epsilon = 1e-8

    exponent = tf.math.log((tf.math.abs(V)+epsilon)/V_ref)/tf.math.log(2.0)

    if n_std == tf.constant(0.0):
        n = n_avg
        I = tf.sign(tf.expand_dims(V, axis=-1)) * V_ref * tf.expand_dims(G, axis=0) * n ** (tf.expand_dims(exponent, axis=-1))
    else:
        n = tf.random.normal(G.get_shape().as_list(), mean=n_avg, stddev=n_std, dtype=tf.float32)
        I = tf.sign(tf.expand_dims(V, axis=-1)) * V_ref * tf.expand_dims(G, axis=0) * tf.expand_dims(n, axis=0) ** (tf.expand_dims(exponent, axis=-1))

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

