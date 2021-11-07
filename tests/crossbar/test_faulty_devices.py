"""
Tests of functions of crossbar.faulty_devices
"""
import pytest
# pylint: disable=missing-function-docstring
import tensorflow as tf
from tests import utils

from crossbar import faulty_devices

random_devices_stuck_testdata = [
    (
        {
            "G": tf.constant(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
            "val": 2.0,
            "prob": 1.0,
        },
        tf.constant(
            [
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        ),
    ),
    (
        {
            "G": tf.constant(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
            "val": 5.0,
            "prob": 0.0,
        },
        tf.constant(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize("args,expected", random_devices_stuck_testdata)
def test_stuck_random_devices_stuck(args, expected):
    result = faulty_devices.random_devices_stuck(**args)
    utils.assert_tf_approx(result, expected)
