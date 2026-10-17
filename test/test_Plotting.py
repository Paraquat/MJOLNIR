import pytest

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data import Cuts, Plotting
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
HKL1, HKL2 = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

@pytest.fixture(scope="module")
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

@pytest.fixture(scope="module")
def result(dataset):
    qx, qy = dataset.gather_q()
    gather = lambda arr_type: np.asarray(dataset.gather(arr_type))
    q1, q2 = dataset.convert_to_qxqy([HKL1, HKL2])
    return Cuts.cut_qe(qx=np.asarray(qx), qy=np.asarray(qy), energy=gather(ArrayType.ENERGY),
                       intensity=gather(ArrayType.INTENSITY),
                       normalization=gather(ArrayType.NORMALIZATION),
                       monitor=gather(ArrayType.MONITOR), q1=q1, q2=q2, width=0.1,
                       min_pixel=0.02, energy_min=-1.0, energy_max=0.8, energy_step=0.1,
                       sample=dataset.sample)

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

def test_plot_cut_qe(dataset, result):
    ax = Plotting.plot_cut_qe(result, sample=dataset.sample, colorbar=True)
    mesh = ax.pmeshs[0]
    assert mesh.get_array().shape == (18, 82) # Energy bins x Q bins
    assert np.ma.allclose(mesh.get_array(),
                          np.asarray(result.data['Int']).reshape(18, 82), masked_equal=True)
    # x runs from 0 at the start of the cut to 1 at its end; y is the lower energy edge
    length = np.linalg.norm(result.parameters['q2'] - result.parameters['q1'])
    assert ax.X[0] == pytest.approx(0.5*0.02/length) # Middle of the first Q bin
    # Bins run one pixel past q2, so that the whole cut is covered
    assert np.all(np.diff(ax.X) > 0) and 1 < ax.X[-1] < 1 + 0.02/length
    assert np.allclose(ax.Y, result.bins['energy'][:-1])
    assert np.allclose(ax.calculatePosition(0), HKL1) and np.allclose(ax.calculatePosition(1), HKL2)
    assert 'BinDistance' in result.data
    assert ax.get_ylabel() == 'E [meV]'
    assert 'Q_h' in ax.get_xlabel()
    ax.set_clim(0, 0.025)
    assert ax.get_clim() == (0, 0.025)
    assert ax.colorbar is not None

def test_plot_options(dataset, result):
    counts = Plotting.plot_cut_qe(result, sample=dataset.sample, counts=True).pmeshs[0]
    assert np.nansum(counts.get_array()) == result.data['intensity'].sum()
    sensitivity = Plotting.plot_cut_qe(result, sample=dataset.sample, counts='sensitivity')
    assert sensitivity.pmeshs[0].get_array().max() > 0
    log = Plotting.plot_cut_qe(result, sample=dataset.sample, log=True).pmeshs[0]
    linear = np.asarray(result.data['Int']).reshape(18, 82)
    assert np.ma.allclose(log.get_array(), np.log10(linear + 1e-20), masked_equal=True)
    in_q = Plotting.plot_cut_qe(result, rlu=False)
    assert 'Q_x' in in_q.get_xlabel()
    with pytest.raises(AttributeError): # H, K, L need the sample
        Plotting.plot_cut_qe(result)
    with pytest.raises(AttributeError):
        Plotting.plot_cut_qe(result, sample=dataset.sample, counts='photons')

def test_given_axes_and_csv(dataset, result, tmp_path):
    _, ax = plt.subplots()
    returned = Plotting.plot_cut_qe(result, sample=dataset.sample, ax=ax)
    assert returned is ax
    ax.to_csv(tmp_path / 'cut.csv')
    lines = (tmp_path / 'cut.csv').read_text().splitlines()
    assert lines[0].startswith('# CSV generated from MJOLNIR')
    assert len(lines) == 2 + len(result) # Comment, header, one line per bin

def test_tick_labels(dataset, result):
    ax = Plotting.plot_cut_qe(result, sample=dataset.sample)
    ax.get_figure().canvas.draw()
    first = ax.get_xticklabels()[0].get_text().split('\n')
    assert len(first) == 3 # H, K and L of the tick
