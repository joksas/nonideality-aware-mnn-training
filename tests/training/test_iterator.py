"""
Tests of functions of training.iterator
"""
# pylint: disable=missing-function-docstring
import pytest
from training import iterator


nonideality_label_testdata = [
        (
            iterator.Nonideal(
                ),
            "ideal",
            ),
        (
            iterator.Nonideal(
                iv_nonlinearity=iterator.IVNonlinearity(1.53, 0.625),
                ),
            "IVNL:1.530_0.625",
            ),
        (
            iterator.Nonideal(
                stuck_at_G_min=iterator.StuckAtGMin(0.6341),
                ),
            "StuckMin:0.634",
            ),
        (
            iterator.Nonideal(
                iv_nonlinearity=iterator.IVNonlinearity(1.530, 0.123),
                stuck_at_G_min=iterator.StuckAtGMin(0.6341),
                ),
            "IVNL:1.530_0.123+StuckMin:0.634",
            ),
        (
            iterator.Nonideal(
                iv_nonlinearity=iterator.IVNonlinearity(3.1, 0.1203),
                stuck_at_G_min=iterator.StuckAtGMin(0.0009),
                stuck_at_G_max=iterator.StuckAtGMax(0.1),
                ),
            "IVNL:3.100_0.120+StuckMin:0.001+StuckMax:0.100",
            ),
        ]


@pytest.mark.parametrize("nonideality_instance,expected", nonideality_label_testdata)
def test_random_bool_tensor(nonideality_instance, expected):
    result = nonideality_instance.nonideality_label()
    assert result == expected
