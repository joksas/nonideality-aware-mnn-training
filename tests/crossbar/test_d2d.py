"""
Tests of functions of crossbar.d2d
"""
# pylint: disable=missing-function-docstring
import tensorflow as tf
import pytest
from crossbar import d2d
from tests import utils


# Only special case, i.e. when std = 0.0 for all entries.
lognormal_testdata = [
    (
        {
            "G": tf.constant(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
            "G_min": 0.5,
            "G_max": 6.0,
            "R_min_std": 0.0,
            "R_max_std": 0.0,
        },
        tf.constant(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    ),
    (
        {
            "G": tf.constant(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            "G_min": 1.0,
            "G_max": 6.0,
            "R_min_std": 1.0,
            "R_max_std": 0.0,
        },
        tf.constant(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize("args,expected", lognormal_testdata)
def test_lognormal(args, expected):
    result = d2d.lognormal(**args)
    utils.assert_tf_approx(result, expected)
