import tensorflow as tf


def assert_tf_approx(a, b):
    tf.debugging.assert_near(a, b, rtol=1.0e-6, atol=1.0e-6)
    assert a.shape == b.shape

