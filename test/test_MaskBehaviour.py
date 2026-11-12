import pytest
import warnings

import numpy as np

from MJOLNIR.Data import Mask

points = np.array([[0.0, 0.5, 1.5, 3.5],
                   [0.0, 0.0, 0.0, 0.0]])

def test_parse():
    # parse must work on every Python version (Python 3.13 changed how locals() works)
    m1 = Mask.circleMask(center=[0, 0], radius=1, name='m1')
    m2 = Mask.circleMask(center=[3, 0], radius=1, name='m2')
    m3 = Mask.circleMask(center=[0, 0], radius=3, name='m3')
    combined = Mask.parse('(m1+m2)*m3', [m1, m2, m3])
    assert isinstance(combined, Mask.MultiMask)
    assert list(combined(points)) == [True, True, False, False]

def test_hash_matches_eq():
    a = Mask.circleMask(center=[0, 0], radius=1)
    b = Mask.circleMask(center=[0, 0], radius=1)
    c = Mask.circleMask(center=[0, 0], radius=2)
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert hash(a) != hash(c) # The old hash (of a temporary object) gave every mask the same
    assert len({a, b, c}) == 2
    assert hash(a + c) == hash(b + c) # Combinations too

def test_subtract_keeps_masks():
    a = Mask.circleMask(center=[0, 0], radius=3)
    b = Mask.circleMask(center=[0, 0], radius=1)
    b_before = list(b(points))
    difference = a - b
    assert not b.negated
    assert list(b(points)) == b_before
    # a - b means a + (-b): in a, or not in b
    assert np.array_equal(difference(points), a(points) | ~b(points))

def test_combining_keeps_names():
    first = Mask.circleMask(center=[0, 0], radius=1)
    second = Mask.circleMask(center=[3, 0], radius=1)
    combined = first + second
    assert first.name == second.name == 'mask_0' # Inputs are not renamed
    assert sorted(combined.getNames()) == ['mask_0', 'mask_1'] # Copies are
    # The same mask used several times keeps one name
    again = combined + second + second
    assert sorted(again.getNames()) == ['mask_0', 'mask_1']
    assert list(again(points)) == [True, True, False, True] # Within 1 of x = 0 or x = 3

def test_box_without_corner4_clones():
    box = Mask.boxMask([0, 0, 0], [1, 2, 0], [0, 0, 3])
    clone = box.clone()
    assert clone == box
    assert np.array_equal(clone.corners, box.corners)
    xyz = np.array([[0.5, 2.0], [1.0, 1.0], [1.0, 1.0]]) # Inside, outside
    assert list(box(xyz)) == [True, False]
    assert list((-box)(xyz)) == [False, True]

def test_rotation_matrix_without_axis():
    assert np.array_equal(Mask.RotationMatrix3D(30), np.eye(3))

def test_combined_plot_skips_parts_without_a_shape():
    """indexMask and CurratAxeMask cannot be drawn: a combination containing them draws its
    other parts and warns, instead of raising"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    circle = Mask.circleMask(center=[0, 0], radius=0.5, name='circle')
    rectangle = Mask.rectangleMask(corner1=[0, 0], corner2=[1, 0], corner3=[1, 1], name='box')
    combined = (circle + Mask.indexMask(0, 3, axis=1, name='first')) * (
        rectangle - Mask.CurratAxeMask([[1, 0, 0]], dqx=0.1, dqy=0.1, name='spurion'))
    ax = plt.figure().add_subplot()
    with pytest.warns(UserWarning, match='indexMask first, CurratAxeMask spurion'):
        combined.plot(ax)
    assert len(ax.lines) == 2 # The circle and the rectangle
    plain = plt.figure().add_subplot()
    with warnings.catch_warnings():
        warnings.simplefilter('error') # Nothing skipped: no warning
        (circle + rectangle).plot(plain)
    assert len(plain.lines) == 2
    with pytest.raises(NotImplementedError): # A single one still says it cannot be drawn
        Mask.indexMask(0, 3).plot(ax)
    plt.close('all')
