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
def test_random_bool_tensor(args, expected):
    result = faulty_devices.random_bool_tensor(**args)
    utils.assert_tf_bool_equal(result, expected)


random_devices_stuck_testdata = [
        (
            {
                "G": tf.constant([
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    ]),
                "val": 2.0,
                "prob": 1.0,
                },
            tf.constant([
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                ])
            ),
        (
            {
                "G": tf.constant([
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    ]),
                "val": 5.0,
                "prob": 0.0,
                },
            tf.constant([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                ])
            ),
        ]

@pytest.mark.parametrize("args,expected", random_devices_stuck_testdata)
def test_stuck_random_devices_stuck(args, expected):
    result = faulty_devices.random_devices_stuck(**args)
    utils.assert_tf_approx(result, expected)
