import subprocess
import sys

import numpy as np
from matplotlib.patches import Circle, Rectangle

from MJOLNIR.Data.DataReader import DataReader
from MJOLNIR.Data.DraggableShapes import center2corner, corner2center, \
    extractCut1DPropertiesCircle, extractCut1DPropertiesRectangle, \
    extractCut1DPropertiesRectangleHorizontal, extractCut1DPropertiesRectanglePerpendicular, \
    extractCut1DPropertiesRectangleVertical

def hhl_sample():
    """Cubic sample (a = 5 Å) in the (H,H,0)-(0,0,L) plane"""
    return DataReader.sample_from_reflections([5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                                              np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))

def test_no_qt_at_import():
    code = ('import sys, MJOLNIR.Data.DraggableShapes; '
            'print(any(m.split(".")[0] in ("PyQt5", "PyQt6", "PySide2", "PySide6", "pyqtgraph") '
            'for m in sys.modules))')
    output = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            check=True).stdout
    assert output.strip() == 'False'

def test_corner_center_inverse():
    corner = center2corner(0.3, -0.2, 0.4, 30.0)
    assert np.allclose(corner2center(corner[0], corner[1], 0.4, 30.0), [0.3, -0.2])

def test_rectangle_qxqy():
    # Width 0.2 along x, height 1.0 along y: the cut runs up the middle
    rect = Rectangle((0.9, 0.0), 0.2, 1.0, angle=0.0)
    params = extractCut1DPropertiesRectangle(rect)
    assert np.allclose(params['q1'], [1.0, 0.0])
    assert np.allclose(params['q2'], [1.0, 1.0])
    assert params['width'] == 0.2
    assert params['rlu'] is False

def test_rectangle_rotated():
    # Rotated about the corner: height along +x, width along -y
    rect = Rectangle((0.0, 0.0), 0.2, 1.0, angle=-90.0)
    params = extractCut1DPropertiesRectangle(rect)
    assert np.allclose(params['q1'], [0.0, -0.1])
    assert np.allclose(params['q2'], [1.0, -0.1])

def test_rectangle_hkl():
    sample = hhl_sample()
    q1 = sample.calculateHKLToQxQy(1.0, 1.0, 0.0)
    q2 = sample.calculateHKLToQxQy(1.0, 1.0, 1.0)
    length = np.linalg.norm(q2-q1)
    angle = np.rad2deg(np.arctan2(*(q2-q1)[::-1]))-90
    rect = Rectangle(center2corner(q1[0], q1[1], 0.1, angle), 0.1, length, angle=angle)
    params = extractCut1DPropertiesRectangle(rect, sample)
    assert np.allclose(params['q1'], [1, 1, 0], atol=1e-3)
    assert np.allclose(params['q2'], [1, 1, 1], atol=1e-3)
    assert params['rlu'] is True

def test_rectangle_parameters_fit_cut_1d():
    params = extractCut1DPropertiesRectangle(Rectangle((0.0, 0.0), 0.1, 1.0), hhl_sample())
    assert set(params) == {'q1', 'q2', 'width', 'min_pixel', 'rlu', 'constant_bins'}

def test_circle():
    sample = hhl_sample()
    center = sample.calculateHKLToQxQy(1.0, 1.0, 0.0)
    params = extractCut1DPropertiesCircle(Circle(center, 0.05), sample)
    assert np.allclose(params['q'], [1, 1, 0], atol=1e-3)
    assert params['width'] == 0.1 # cut_constant_q's width is the diameter
    assert set(params) == {'q', 'width', 'rlu', 'constant_bins'}

def test_qe_rectangles():
    # x along the Q-E cut from 0.2 to 0.4, energy 5 down to 3 (drawn downwards)
    rect = Rectangle((0.2, 5.0), 0.2, -2.0)
    perpendicular = extractCut1DPropertiesRectanglePerpendicular(rect)
    assert np.allclose(perpendicular['center'], [0.3, 4.0])
    assert perpendicular['width'] == 0.2
    for params in (perpendicular, extractCut1DPropertiesRectangleHorizontal(rect),
                   extractCut1DPropertiesRectangleVertical(rect)):
        assert (params['energy_min'], params['energy_max']) == (3.0, 5.0)
    horizontal = extractCut1DPropertiesRectangleHorizontal(rect)
    assert np.allclose([horizontal['q1'][0], horizontal['q2'][0]], [0.2, 0.4])
    assert np.isclose(extractCut1DPropertiesRectangleVertical(rect)['q'], 0.3)
