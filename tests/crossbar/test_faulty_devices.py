"""
Tests of functions of crossbar.faulty_devices
"""
# pylint: disable=missing-function-docstring
import tensorflow as tf
import pytest
from crossbar import faulty_devices
from tests import utils


random_bool_tensor_testdata = [
        (
            {
                "shape": [2, 3],
                "prob_true": 0.0,
                },
            tf.constant([
                [False, False, False],
                [False, False, False],
                ])
            ),
        (
            {
                "shape": [4],
                "prob_true": 1.0,
                },
            tf.constant([True, True, True, True])
            ),
        ]


@pytest.mark.parametrize("args,expected", random_bool_tensor_testdata)
def test_compute_currents(args, expected):
    result = faulty_devices.random_bool_tensor(**args)
    utils.assert_tf_bool_equal(result, expected)
