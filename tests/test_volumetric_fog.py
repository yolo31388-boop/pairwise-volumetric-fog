import pytest

from render.volumetric_fog import (
    froxel_depth,
    integrate,
    reproject,
    shadow_slice,
    composite
)


def test_froxel_near():
    assert froxel_depth(0, 0.1, 100.0, 8, 0) == 0.1

def test_froxel_far():
    assert froxel_depth(7, 0.1, 100.0, 8, 7) == 100.0

def test_froxel_monotonic():
    assert froxel_depth(2, 0.1, 100.0, 8, 2) > froxel_depth(1, 0.1, 100.0, 8, 1)

def test_integrate_zero():
    assert integrate(0.0, 1.0, 4) == 0.0

def test_integrate_positive():
    assert integrate(1.0, 1.0, 4) > 0.0

def test_integrate_density():
    assert integrate(2.0, 1.0, 4) > integrate(1.0, 1.0, 4)

def test_reproject_same():
    assert reproject([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], (0.0, 0.0)) == [1.0, 2.0, 3.0]

def test_reproject_shift():
    assert reproject([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], (0.1, 0.0)) != [1.0, 2.0, 3.0]

def test_shadow_first():
    assert shadow_slice(0) == 0

def test_shadow_last():
    assert shadow_slice(1.0) == 1

def test_composite_clear():
    assert composite((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0) == (0.0, 0.0, 0.0)

def test_composite_full():
    assert composite((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0) == (1.0, 1.0, 1.0)

def test_composite_half():
    assert composite((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.5) == pytest.approx((0.5, 0.5, 0.5))

def test_composite_alpha0():
    assert composite((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0) == (1.0, 1.0, 1.0)

def test_integrate_steps():
    assert integrate(1.0, 1.0, 1) > 0.0
