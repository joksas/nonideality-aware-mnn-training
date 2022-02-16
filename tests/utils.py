import numpy as np
import tensorflow as tf


def assert_tf_approx(a, b, tol=1.0e-6):
    tf.debugging.assert_near(a, b, rtol=tol, atol=tol)
    assert a.shape == b.shape


def assert_tf_bool_equal(a, b):
    # Don't know how to compare boolean arrays using TF, so using numpy.
    np.testing.assert_array_equal(a, b)
