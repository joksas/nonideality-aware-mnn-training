from scipy import interpolate
import numpy as np
import copy
from badmemristor_tf.nonideality import utils
import tensorflow as tf


def compute_I(V, G, V_ref, G_min, G_max, n_avg=None, n_std=tf.constant(0.0), eff=False, model="lookup_table"):
    """Computes output currents of a crossbar consisting of devices suffering
    from I/V non-linearities.

    Parameters
    ----------
    V : ndarray
        Voltages of shape (p x m).
    G : ndarray
        Conductances (or effective conductances) of shape (m x n).
    V_ref :
        Reference voltage values of length r (in increasing order) or voltage
        at which the devices behave Ohmically.
    G_min : float
        Minimum conductance of electroformed memristors.
    G_max : float
        Maximum conductance of electroformed memristors.
    I_ref :
        Reference current values of shape (q x r) corresponding go G_ref and
        V_ref.
    n_avg : tf.constant, optional
        Average value of non-linearity parameter.
    n_std: tf.constant, optional
        Standard deviation of non-linearity parameter.
    eff : bool, optional
        If True, it means that effective conductances have been passed.
    model : {"lookup_table", "nonlinear_param"}, optional
        The model used for computing non-linear behaviour of current.

    Returns
    ----------
    I : ndarray
        Output currents of shape (p x n). If eff is True, then *total* output
        currents are returned.
    """
    if model == "lookup_table":
        I_ind = interpolate_I(G_ref, V_ref, I_ref, G, V, eff)
    elif model == "nonlinear_param":
        I_ind = interpolate_I_nonlinear_param(G_min, G_max, n_avg, V_ref, G, V, n_std=n_std)

    I = add_I_BL(I_ind)

    return I


def interpolate_I(G_ref, V_ref, I_ref, G, V, eff):
    """Interpolates current values.

    Parameters
    ----------
    G_ref : ndarray
        Reference conductance values of length q (in increasing order).
    V_ref : ndarray
        Reference voltage values of length r (in increasing order).
    I_ref : ndarray
        Reference voltage values of shape (q x r) corresponding go G_ref and
        V_ref.
    G : ndarray
        Conductances (or effective conductances) of shape (m x n).
    V : ndarray
        Voltages of shape (p x m).
    eff : bool
        If True, it means that effective conductances have been passed.

    Returns
    ----------
    I : ndarray
        Interpolated currents of shape (p x m x n) produced by each of the
        conductances in the crossbar array.
    """
    if eff:
        if G_ref[0] == 0:
            # If the lowest conductance state in G_ref is 0 S, then we must
            # make sure that it isn't duplicated.
            G_ref = np.concatenate((-G_ref[::-1], G_ref[1:]), axis=0)
            I_ref = np.concatenate((-I_ref[::-1, :], I_ref[1:]), axis=0)
        else:
            G_ref = np.concatenate((-G_ref[::-1], G_ref), axis=0)
            I_ref = np.concatenate((-I_ref[::-1, :], I_ref), axis=0)

    f = interpolate.interp2d(V_ref, G_ref, I_ref, kind='linear')

    num_WL = G.shape[0]

    I_seq = []
    for i_WL in range(num_WL):
        G_slice = G[i_WL, :]
        V_slice = V[:, i_WL]

        # For some reason, scipy interpolation returns results in increasing
        # order of the arguments. Therefore, we must make a note of the order.
        G_slice_idx = np.argsort(G_slice)
        V_slice_idx = np.argsort(V_slice)

        # Interpolation.
        I_slice_wrong = f(V_slice, G_slice)

        # Restoring original order.
        I_slice_wrong_V = copy.deepcopy(I_slice_wrong)
        I_slice_wrong_V[G_slice_idx, :] = I_slice_wrong
        I_slice = copy.deepcopy(I_slice_wrong_V)
        I_slice[:, V_slice_idx] = I_slice_wrong_V

        I_seq.append(I_slice)

    I = np.stack(I_seq, axis=2)
    I = np.swapaxes(I, 0, 2)
    I = np.swapaxes(I, 0, 1)

    return I


def interpolate_I_nonlinear_param(G_min, G_max, n_avg, V_ref, G, V, n_std=tf.constant(0.0)):
    """Interpolates current values.

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
        Conductances (or effective conductances) of shape (m x n).
    V : ndarray
        Voltages of shape (p x m).
    n_std: tf.constant, optional
        Standard deviation of non-linearity parameter.

    Returns
    ----------
    I : ndarray
        Interpolated currents of shape (p x m x n) produced by each of the
        conductances in the crossbar array.
    """
    epsilon = 1e-4

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
        Currents of shape (p x m x n) produced by each of the conductances in
        the crossbar array.

    Returns
    ----------
    I : ndarray
        Output currents of shape (p x n).
    """
    I = tf.math.reduce_sum(I_ind, axis=1)
    return I

