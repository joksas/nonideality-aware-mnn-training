"""
Tests of functions of training.iterator
"""
# pylint: disable=missing-function-docstring
import pytest
from crossbar import nonidealities

from training import iterator

nonideality_label_testdata = [
    (
        iterator.Nonideal(),
        "ideal",
    ),
    (
        iterator.Nonideal(
            nonidealities=[nonidealities.IVNonlinearity(1.53, 0.625)],
        ),
        "IVNL:1.53_0.625",
    ),
    (
        iterator.Nonideal(
            nonidealities=[nonidealities.StuckAt(1.20, 0.6341)],
        ),
        "Stuck:1.2_0.634",
    ),
    (
        iterator.Nonideal(
            nonidealities=[
                nonidealities.IVNonlinearity(1.530, 0.123),
                nonidealities.StuckAt(1.2344, 0.06341),
            ]
        ),
        "IVNL:1.53_0.123+Stuck:1.23_0.0634",
    ),
    (
        iterator.Nonideal(
            nonidealities=[
                nonidealities.IVNonlinearity(3.1, 0.1203),
                nonidealities.StuckAt(1.23, 0.0009),
                nonidealities.StuckAt(4.5, 0.1),
            ]
        ),
        "IVNL:3.1_0.12+Stuck:1.23_0.0009+Stuck:4.5_0.1",
    ),
]


@pytest.mark.parametrize("nonideal_instance,expected", nonideality_label_testdata)
def test_random_bool_tensor(nonideal_instance, expected):
    result = nonideal_instance.nonideality_label()
    assert result == expected
