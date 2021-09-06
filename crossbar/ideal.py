import tensorflow as tf
from . import utils


def compute_I_all(V, G):
    I_ind = tf.expand_dims(V, axis=-1) * tf.expand_dims(G, axis=0)
    I = utils.add_I_BL(I_ind)
    return I, I_ind


def compute_I(V, G):
    return tf.tensordot(V, G, axes=1)
