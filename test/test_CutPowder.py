import pytest

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data import Cuts, Plotting
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')
Q_BINS, ENERGY_BINS = np.arange(0.4, 2.0, 0.05), np.arange(-1, 0.81, 0.1)

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

def toy_points(**extra):
    """|Q| = 1, 1, 2, 5 (3-4-5 triangle); energies 0.5, 0.5, 1.5, 1.5"""
    points = dict(qx=np.array([1.0, 0.0, 2.0, 3.0]), qy=np.array([0.0, -1.0, 0.0, 4.0]),
                  energy=np.array([0.5, 0.5, 1.5, 1.5]),
                  intensity=np.array([4.0, 9.0, 16.0, 25.0]),
                  normalization=np.full(4, 2.0), monitor=np.full(4, 10.0))
    points.update(extra)
    return points

def test_cut_powder():
    result = Cuts.cut_powder(**toy_points(), q_bins=[0.0, 1.5, 3.0], energy_bins=[0.0, 1.0, 2.0])
    data = result.data
    assert list(data['q']) == [0.75, 0.75, 2.25, 2.25] # |Q| bin, then energy bin
    assert list(data['Energy']) == [0.5, 1.5, 0.5, 1.5]
    assert list(data['intensity']) == [13.0, 0.0, 0.0, 16.0] # |Q| = 5 is outside
    assert list(data['BinCount']) == [2, 0, 0, 1]
    assert data['Int'].iloc[0] == pytest.approx(13*2/(20*4))
    assert data['Int_err'].iloc[0] == pytest.approx(np.sqrt(13)*2/(20*4))
    assert np.isnan(data['Int'].iloc[1]) # Empty bin
    assert result.grid('intensity').tolist() == [[13.0, 0.0], [0.0, 16.0]]
    # The last edges are included
    edge = Cuts.cut_powder(**toy_points(), q_bins=[0.0, 1.0, 5.0], energy_bins=[0.0, 1.5])
    assert list(edge.data['BinCount']) == [0, 4]

def test_cut_powder_errors_and_background():
    sigma = np.array([1.0, 2.0, 3.0, 4.0])
    background = np.array([1.0, 1.0, 2.0, 2.0])
    data = Cuts.cut_powder(**toy_points(intensity_error=sigma, background=background),
                           q_bins=[0.0, 1.5, 3.0], energy_bins=[0.0, 1.0, 2.0]).data
    scale = 2/(20*4)
    assert data['intensity'].iloc[0] == 13.0 # The measured counts
    assert (data['intensity_Fg'].iloc[0], data['intensity_Bg'].iloc[0]) == (13.0, 2.0)
    assert data['Int'].iloc[0] == pytest.approx((13 - 2)*scale)
    assert data['Int_Fg'].iloc[0] == pytest.approx(13*scale)
    assert data['Int_err'].iloc[0] == pytest.approx(np.sqrt(1 + 4 + 2)*scale)

def test_cut_powder_bins_checked():
    for q_bins, energy_bins in [([1.0], [0.0, 1.0]), ([0.0, 1.0], [1.0, 0.0]),
                                ([[0.0, 1.0]], [0.0, 1.0])]:
        with pytest.raises(ValueError, match='increasing bin edges'):
            Cuts.cut_powder(**toy_points(), q_bins=q_bins, energy_bins=energy_bins)

@pytest.fixture(scope="module")
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

@needs_data
def test_dataset_cut_powder_background_and_errors(dataset):
    plain = dataset.cut_powder(Q_BINS, ENERGY_BINS).data
    dataset.generate_background_model()
    try:
        data = dataset.cut_powder(Q_BINS, ENERGY_BINS, background_subtraction=True).data
    finally:
        dataset.remove_background_model()
    assert np.array_equal(data['intensity'], plain['intensity'])
    scale = data['BinCount']/(data['Monitor']*data['Normalization'])
    assert np.allclose(data['Int'], (data['intensity_Fg'] - data['intensity_Bg'])*scale,
                       equal_nan=True)
    assert np.allclose(data['Int_err'],
                       np.sqrt(data['intensity_Fg'] + data['intensity_Bg'])*scale, equal_nan=True)
    difference = (dataset - dataset).cut_powder(Q_BINS, ENERGY_BINS).data
    assert np.allclose(difference['Int_err'], np.sqrt(2)*plain['Int_err'], equal_nan=True)

@needs_data
def test_plot_cut_powder(dataset, tmp_path):
    result = dataset.cut_powder(Q_BINS, ENERGY_BINS)
    ax = Plotting.plot_cut_powder(result)
    mesh = ax.pmeshs[0]
    shape = (len(ENERGY_BINS) - 1, len(Q_BINS) - 1)
    assert np.ma.allequal(mesh.get_array().reshape(shape),
                          np.ma.masked_invalid(result.grid('Int')).T)
    coordinates = mesh.get_coordinates() # Drawn between the bin edges
    assert np.allclose(coordinates[0, :, 0], Q_BINS) and np.allclose(coordinates[:, 0, 1], ENERGY_BINS)
    finite = result.data['Int'].dropna()
    assert ax.get_clim() == (finite.min(), finite.max()) # get_clim crashed in the old plot
    assert ax.get_xlabel() == r'$|q| [\AA^{-1}]$' and ax.get_ylabel() == 'E [meV]'
    assert ax.colorbar is not None and ax.type == 'Powder'
    q_index, e_index = ax.calculateIndex(Q_BINS[3] + 0.01, ENERGY_BINS[5] + 0.01)
    assert (q_index, e_index) == (3, 5)
    assert ax.calculateIndex(10.0, 0.0) == (-1, 10)
    ax.to_csv(tmp_path / 'powder.csv')
    assert len((tmp_path / 'powder.csv').read_text().splitlines()) == 2 + len(result)

@needs_data
def test_plot_cut_powder_options(dataset):
    ax = dataset.plot_cut_powder(Q_BINS, ENERGY_BINS, log=True, vmin=-8, vmax=-2,
                                 colorbar=False)
    assert ax.get_clim() == (-8, -2) and not hasattr(ax, 'colorbar')
    counts = dataset.plot_cut_powder(Q_BINS, ENERGY_BINS, counts=True, log=True)
    assert (ax.result.data['intensity'] == 0).any() # Zero counts: log10(0 + 1e-20), not -inf
    assert np.isfinite(counts.pmeshs[0].get_array().compressed()).all()
    assert counts.get_clim()[0] == pytest.approx(-20)
    with pytest.raises(ValueError, match='cut_powder'):
        Plotting.plot_cut_powder(dataset.cut_1d([-1, 0, 0], [0, 1, 0], 0.1, 0.02, -1.0, 0.8))
