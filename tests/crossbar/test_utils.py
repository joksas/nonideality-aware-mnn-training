"""
Tests of functions of crossbar.faulty_devices
"""
import pytest
import tensorflow as tf

from awarememristor.crossbar import utils as crossbar_utils
from tests import utils

random_bool_tensor_testdata = [
    (
        {
            "shape": [2, 3],
            "prob_true": 0.0,
        },
        tf.constant(
            [
                [False, False, False],
                [False, False, False],
            ]
        ),
    ),
    (
        {
            "shape": [4],
            "prob_true": 1.0,
        },
        tf.constant([True, True, True, True]),
    ),
]


@pytest.mark.parametrize("args,expected", random_bool_tensor_testdata)
def test_random_bool_tensor(args, expected):
    result = crossbar_utils.random_bool_tensor(**args)
    utils.assert_tf_bool_equal(result, expected)
