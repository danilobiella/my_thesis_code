"""
Tests for functions in the functions module
"""

import numpy as np
import pytest
import src.utils as f
from src.lightcurve import LightCurve, LightCurveCycles, cut_lc


def test_find_isolated_local_maxima_base():
    """Test find_isolated_local_maxima in expected periodic input"""
    input_array = [0.0, 1, 3, 4, 5, 4, 3, 2, 1, 0, 3, 4, 10, 6, 3, 2, 1, 4, 5]
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(input_array, 5), [4, 12, 18]
    )
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(input_array, 10), [1, 12]
    )
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(input_array, 3), [0, 4, 8, 12, 18]
    )

    # Case with same value too close to maximum
    input_array = [0.0, 1, 3, 4, 5, 4, 3, 2, 1, 0, 10, 0, 10, 6, 3, 2, 1, 4, 5]
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(input_array, 5), ([4, 10, 18])
    )


def test_find_isolated_local_maxima_edge_cases():
    """Test find_isolated_local_maxima edge cases"""

    # Case empty list
    np.testing.assert_array_equal(f.find_isolated_local_maxima(([]), 1), ([]))
    np.testing.assert_array_equal(f.find_isolated_local_maxima(([]), 100), ([]))

    # Case zero max_distance
    np.testing.assert_array_equal(f.find_isolated_local_maxima(([]), 0), ([]))
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(([10, 4, 3, 4]), 0),
        ([0, 1, 2, 3]),
    )

    # Case max_distance is greater than length of array
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(([1, 2, 3, 2, 1]), 100), ([2])
    )
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(([1, 2, 3, 2, 1, 2, 3, 4, 3]), 100),
        ([7]),
    )

    # One element array
    np.testing.assert_array_equal(f.find_isolated_local_maxima(([5]), 10), ([0]))
    np.testing.assert_array_equal(f.find_isolated_local_maxima(([5]), 1), ([0]))

    # Case monotonic array
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(([10, 20, 30, 40, 50, 60]), 100),
        ([5]),
    )
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(([10, 20, 30, 40, 50, 60]), 2),
        ([2, 5]),
    )

    # Case constant array"""
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(([1, 1, 1, 1]), 100), ([0])
    )
    np.testing.assert_array_equal(
        f.find_isolated_local_maxima(([1, 1, 1, 1]), 1), ([0, 2])
    )


def test_find_isolated_local_maxima_raises():
    """Test find_isolated_local_maxima raises proper exceptions"""
    # Negative min_distance
    with pytest.raises(ValueError):
        f.find_isolated_local_maxima(([0, 0, 1, 2, 3, 0, 1, 2]), -10)
    with pytest.raises(ValueError):
        f.find_isolated_local_maxima(([]), -10)


def test_compute_periods_base():
    """Test cut_lc in expected periodic input"""
    np.testing.assert_array_equal(
        f.compute_periods(np.array([5, 10, 14, 20])), np.array([5, 4, 6])
    )
    np.testing.assert_array_equal(
        f.compute_periods(np.array([0, 5, 10, 14, 20])), np.array([5, 5, 4, 6])
    )


def test_compute_periods_edge_cases():
    # Empty list
    np.testing.assert_array_equal(f.compute_periods(np.array([])), np.array([]))

    # One element list
    np.testing.assert_array_equal(f.compute_periods(np.array([10])), np.array([]))

    # 1 period list
    np.testing.assert_array_equal(
        f.compute_periods(np.array([0, 1, 2, 3])), np.array([1, 1, 1])
    )

    # with zero period
    np.testing.assert_array_equal(
        f.compute_periods(np.array([5, 10, 10, 13])), np.array([5, 0, 3])
    )


def test_cut_lc_base():
    """Test cut_lc in expected periodic input"""
    lc = LightCurve(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 20, 30, 40, 10, 20, 30, 40, 10, 20]
    )
    cycle_bounds = [4, 6, 8]
    expected = LightCurveCycles(
        [
            LightCurve([4, 5], [10, 20]),
            LightCurve([6, 7], [30, 40]),
        ]
    )

    assert cut_lc(lc, cycle_bounds) == expected

    lc = LightCurve(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 20, 30, 40, 10, 20, 30, 40, 10, 20]
    )
    cycle_bounds = [0, 4, 8, 10]
    expected = LightCurveCycles(
        [
            LightCurve([0, 1, 2, 3], [10, 20, 30, 40]),
            LightCurve([4, 5, 6, 7], [10, 20, 30, 40]),
            LightCurve([8, 9], [10, 20]),
        ]
    )

    assert cut_lc(lc, cycle_bounds) == expected


def test_make_template_base():
    lc_cycles = [
        [10.0, 20, 30, 40],
        [10.0, 20],
    ]

    result = f.make_template(lc_cycles, n_bins=2)
    expected = np.array([(10 + 20 + 10) / 3, (40 + 30 + 20) / 3])
    np.testing.assert_array_equal(result, expected)

    # Varying length
    lc_cycles = [
        [9.0, 15, 20, 25, 30, 35],
        [10.0, 20, 30, 44],
        [10.0, 40],
    ]

    result = f.make_template(lc_cycles, n_bins=2)
    expected = np.array([14, 34])
    np.testing.assert_array_equal(result, expected)

    lc_cycles = [
        [10.0, 15, 21, 25, 30],
        [10.0, 20, 30, 40],
        [10.0, 40, 50],
    ]

    result = f.make_template(lc_cycles, n_bins=3)
    expected = np.array([11.25, 27, 35])
    np.testing.assert_array_equal(result, expected)


def test_make_template_edge_cases():
    # 1 bin
    lc_cycles = [
        [10.0, 15, 20, 25, 30],
        [10.0, 20, 30, 40],
        [10.0, 40, 50],
    ]

    result = f.make_template(lc_cycles, n_bins=1)
    expected = np.mean([10, 15, 20, 25, 30, 10, 20, 30, 40, 10, 40, 50])
    np.testing.assert_array_equal(result, expected)

    # Empty cycles
    lc_cycles = []
    result = f.make_template(lc_cycles, n_bins=10)
    expected = np.empty(0)
    np.testing.assert_array_equal(result, expected)


def test_make_template_raises():
    lc_cycles = [
        [10.0, 15, 20, 25, 30, 35],
        [10.0, 20, 30, 40],
        [10.0, 40],
    ]

    # Invalid n_bins
    with pytest.raises(ValueError):
        f.make_template(lc_cycles, n_bins=0)
    with pytest.raises(ValueError):
        f.make_template(lc_cycles, n_bins=-10)


def test_stretch_base():

    # Constant cases
    result = f.stretch([10, 10, 10], 5)
    expected = [10, 10, 10, 10, 10]
    np.testing.assert_array_equal(expected, result)

    result = f.stretch([10, 100], 2)
    expected = [10, 100]
    np.testing.assert_array_equal(expected, result)

    result = f.stretch([10, 20, 30], 5)
    expected = [10, 15, 20, 25, 30]
    np.testing.assert_array_equal(expected, result)

    result = f.stretch([10, 20, 30], 4)
    expected = [10, 10 + 20 / 3, 10 + 20 / 3 * 2, 30]
    np.testing.assert_allclose(expected, result)

    result = f.stretch([10, 80], 8)
    expected = [10, 20, 30, 40, 50, 60, 70, 80]

    # Smaller length case
    result = f.stretch([10, 20, 30, 40, 50, 60, 70, 80], 2)
    expected = [10, 80]

    # More complex input
    result = f.stretch([10, 30, 10, -10, 0, 100], 11)
    expected = [10, 20, 30, 20, 10, 0, -10, -5, 0, 50, 100]


def test_stretch_edge_cases():

    # Output lengt zero
    result = f.stretch([6, 66, 666], 0)
    expected = []
    np.testing.assert_array_equal(expected, result)

    # Output lengt one
    result = f.stretch([10, 10, 10], 1)
    expected = [10]
    np.testing.assert_array_equal(expected, result)

    result = f.stretch([1, 9, 20], 1)
    expected = [1]
    np.testing.assert_array_equal(expected, result)


def test_stretch_errors():

    # Input length zero
    with pytest.raises(ValueError):
        f.stretch([], 2)


def test_remove_bad_bounds():
    # Don't do anything
    bounds = [10, 20, 30, 40]
    expected = [10, 20, 30, 40]
    np.testing.assert_array_equal(f.remove_bad_bounds(bounds, 8), expected)

    # One bad
    bounds = [10, 25, 30, 40]
    expected = [10, 25, 40]
    np.testing.assert_array_equal(f.remove_bad_bounds(bounds, 8), expected)

    # Two consecutive bad
    bounds = [20, 25, 30, 40]
    expected = [20, 30, 40]
    np.testing.assert_array_equal(f.remove_bad_bounds(bounds, 8), expected)

    # Many bad
    bounds = [20, 25, 30, 40, 45, 50]
    expected = [20, 30, 40, 50]
    np.testing.assert_array_equal(f.remove_bad_bounds(bounds, 8), expected)

    # Many bad in succession
    bounds = [10, 15, 17, 19, 20, 30, 40, 45, 50]
    expected = [10, 19, 30, 40, 50]
    np.testing.assert_array_equal(f.remove_bad_bounds(bounds, 8), expected)


def test_remove_bad_raises():
    # Don't do anything
    with pytest.raises(ValueError):
        f.remove_bad_bounds([], 8)

    with pytest.raises(ValueError):
        f.remove_bad_bounds([10, 20], -10)

    with pytest.raises(ValueError):
        f.remove_bad_bounds([10, 20], 0)


"""
def test_compute_new_bounds_base():
    lc = np.array([0, 0, 0.0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    bounds = np.array([2, 5, 8, 11, 15])
    template = [0.0, 1, 2]
    expected = [4, 8, 12, 16, 20]
    result = f.compute_new_bounds(
        bounds, template, lc, min_period=0, delta_space_size=2
    )
    np.testing.assert_array_equal(expected, result)
"""
