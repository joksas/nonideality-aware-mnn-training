import tensorflow as tf
import numpy as np


def assert_tf_approx(a, b):
    tf.debugging.assert_near(a, b, rtol=1.0e-6, atol=1.0e-6)
    assert a.shape == b.shape


def assert_tf_bool_equal(a, b):
    # Don't know how to compare boolean arrays using TF, so using numpy.
    np.testing.assert_array_equal(a, b)
