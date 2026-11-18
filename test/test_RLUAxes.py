import pytest

import re
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data import RLUAxes
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.Coordinates import align_to_lattice


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

@pytest.fixture(scope='module')
def sample(camea_2024_files):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return CameaDataReader(camea_2024_files[0]).sample

def unsigned_zeros(text: str) -> str:
    """text with '-0.000' as '0.000': the sign of a value that is zero to rounding is noise"""
    return re.sub(r'-(0\.0+)(?!\d*[1-9])', r'\1', text)

def test_q_axis(sample):
    ax = RLUAxes.create_q_axis(sample)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('1, 0, 0 [RLU]', '0, 1, 0 [RLU]')
    aligned = align_to_lattice(sample) # The axes' frame
    assert np.allclose(ax.sample.orientationMatrix, aligned.orientationMatrix)
    assert ax.sample is not sample and sample.theta != 0.0 # The given sample is not changed
    qx, qy = aligned.calculateHKLToQxQy(1.0, 0.0, 0.0)
    # Read-out in H, K, L; k is zero to rounding, so its sign is not checked
    assert unsigned_zeros(ax.format_coord(qx, qy)).startswith('h = 1.000, k = 0.000')
    ax.set_axis([0, 0, 0], [1, 1, 0])
    ax.set_xticks_base(0.25)
    ax.set_yticks_number(5)
    ax.get_figure().canvas.draw() # The grid and the zoom callbacks run
    assert (ax.xbase, ax.yticks) == (0.25, 5)
    with pytest.raises(AttributeError, match='sample is needed'):
        RLUAxes.create_q_axis(None)

def test_q_axis_without_lattice():
    figure = plt.figure()
    ax = RLUAxes.create_q_axis(None, rlu=False, figure=figure, ids=(1, 2, 2))
    assert ax.get_xlabel() == 'Qx [1/AA]' and ax.format_coord(0.3, 0.2) == 'qx = 0.300, qy = 0.200'
    assert ax.get_subplotspec().colspan.start == 1 # ids are used (the old code put it at 111)

def test_qe_axes(sample):
    ax = RLUAxes.create_qe_axes(sample, axis=1)
    assert unsigned_zeros(ax.get_xlabel()) == '0.000, 1.000, 0.000 [RLU]'
    assert ax.get_ylabel() == 'E [meV]'
    # Projection vectors without a sample (the old createQEAxes needed a data set anyway)
    ax = RLUAxes.create_qe_axes(projection_vector1=[1, 1, 0], projection_vector2=[0, 0, 1])
    assert ax.get_xlabel() == '1.000, 1.000, 0.000 [RLU]'
    ax.set_xticks_number(4)
    ax.get_figure().canvas.draw()
    with pytest.raises(AttributeError, match='either 0 or 1'):
        RLUAxes.create_qe_axes(sample, axis=2)
    with pytest.raises(AttributeError, match='or both projection vectors'):
        RLUAxes.create_qe_axes()
