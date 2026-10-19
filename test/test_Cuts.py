import pytest

import numpy as np
import pandas as pd
from pathlib import Path

from MJOLNIR.Data import Cuts
from MJOLNIR.Data.CutResult import CutResult, COLUMNS
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))

# Reference values below were checked against the old DataSet.cut1D / cutQE on the same data
# (identical, except Int_err of cut1D, which the old code truncated to whole counts)

@pytest.fixture(scope="module")
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

@pytest.fixture(scope="module")
def points(dataset):
    qx, qy = dataset.gather_q()
    gather = lambda arr_type: np.asarray(dataset.gather(arr_type))
    return dict(qx=np.asarray(qx), qy=np.asarray(qy), energy=gather(ArrayType.ENERGY),
                intensity=gather(ArrayType.INTENSITY),
                normalization=gather(ArrayType.NORMALIZATION),
                monitor=gather(ArrayType.MONITOR))

@pytest.fixture(scope="module")
def q1q2(dataset):
    return dataset.convert_to_qxqy([[-1, 0, 0], [0, 1, 0]])

def test_cut_1d(dataset, points, q1q2):
    result = Cuts.cut_1d(**points, q1=q1q2[0], q2=q1q2[1], width=0.1, min_pixel=0.02,
                         energy_min=-1.0, energy_max=0.8, sample=dataset.sample)
    assert isinstance(result, CutResult)
    data = result.data
    assert len(result) == 88
    assert list(data.columns) == [COLUMNS[key] for key in ['qx', 'qy', 'h', 'k', 'l', 'e',
        'intensity', 'mon', 'norm', 'binCount', 'intError', 'int']]
    assert data['intensity'].sum() == 187234
    assert data['BinCount'].sum() == 86501
    assert data['BinCount'].dtype == int
    peak = np.nanargmax(data['Int'])
    assert data['Int'].iloc[peak] == pytest.approx(6.422700, rel=1e-6)
    assert data['H'].iloc[peak] == pytest.approx(-1.001195, abs=1e-6)
    assert np.allclose(data['Energy'], -0.1) # The middle of the energy range
    # Normalised intensity and its error from the counting statistics
    scale = data['BinCount']/(data['Normalization']*data['Monitor'])
    assert np.allclose(data['Int'], data['intensity']*scale, equal_nan=True)
    assert np.allclose(data['Int_err'], np.sqrt(data['intensity'])*scale, equal_nan=True)
    assert result.bins['positions'].shape == (89, 3)
    assert result.bins['orthogonal'].shape == (2, 2)
    assert result.parameters['width'] == 0.1

def test_cut_1d_options(points, q1q2):
    common = dict(**points, q1=q1q2[0], q2=q1q2[1], width=0.1, min_pixel=0.02,
                  energy_min=-1.0, energy_max=0.8)
    assert len(Cuts.cut_1d(**common, constant_bins=True)) == 81
    assert len(Cuts.cut_1d(**common, extend=True)) == 152
    # Without a sample there is no H, K, L
    assert Cuts.cut_1d(**common).data['H'].isna().all()
    # No points in the energy range
    empty = Cuts.cut_1d(**{**common, 'energy_min': 50, 'energy_max': 60})
    assert len(empty) == 0
    with pytest.raises(AttributeError): # q1 == q2
        Cuts.cut_1d(**{**common, 'q2': q1q2[0]})

def test_cut_qe(dataset, points, q1q2):
    result = Cuts.cut_qe(**points, q1=q1q2[0], q2=q1q2[1], width=0.1, min_pixel=0.02,
                         energy_min=-1.0, energy_max=0.8, energy_step=0.1,
                         sample=dataset.sample)
    data = result.data
    distance, energy = result.bins['distance'], result.bins['energy']
    assert (len(distance), len(energy)) == (83, 19)
    assert len(result) == 82 * 18 # One row per Q bin and energy bin
    assert list(data.columns) == [COLUMNS[key] for key in ['qx', 'qy', 'h', 'k', 'l', 'e',
        'int', 'intError', 'intensity', 'mon', 'norm', 'binCount']]
    assert data['intensity'].sum() == 137092
    assert data['BinCount'].sum() == 81124
    assert np.nansum(data['Int']) == pytest.approx(516.756264, rel=1e-6)
    # Rows go through the Q bins for each energy bin; Energy is the lower edge of the bin
    assert np.array_equal(np.unique(data['Energy']), energy[:-1])
    assert np.allclose(data['Energy'].iloc[:82], energy[0])
    assert result.bins['orthogonal_extent'].shape == (2, 2)

def test_cut_qe_energy_bins(points, q1q2):
    common = dict(**points, q1=q1q2[0], q2=q1q2[1], width=0.1, min_pixel=0.02)
    by_step = Cuts.cut_qe(**common, energy_min=-1.0, energy_max=0.8, energy_step=0.1)
    by_bins = Cuts.cut_qe(**common, energy_bins=by_step.bins['energy'])
    pd.testing.assert_frame_equal(by_step.data, by_bins.data)
    with pytest.raises(AttributeError): # Neither bins nor min, max and step
        Cuts.cut_qe(**common, energy_min=-1.0, energy_max=0.8)

# Constant-energy planes (reference values checked against the old plotQPlane: identical)

@pytest.fixture(scope="module")
def aligned_points(dataset):
    qx, qy = dataset.gather_q(align_to_lattice=True)
    gather = lambda arr_type: np.asarray(dataset.gather(arr_type))
    return dict(qx=np.asarray(qx), qy=np.asarray(qy), energy=gather(ArrayType.ENERGY),
                intensity=gather(ArrayType.INTENSITY),
                normalization=gather(ArrayType.NORMALIZATION),
                monitor=gather(ArrayType.MONITOR))

def test_cut_q_plane(dataset, aligned_points):
    from MJOLNIR.Data.Coordinates import align_to_lattice
    result = Cuts.cut_q_plane(**aligned_points, energy_min=-0.2, energy_max=0.2,
                              x_bin_tolerance=0.05, y_bin_tolerance=0.05,
                              sample=align_to_lattice(dataset.sample))
    data = result.data
    assert [len(edges) for edges in result.bins['qx']] == [50]
    assert [len(edges) for edges in result.bins['qy']] == [44]
    assert len(result) == 49 * 43
    assert data['intensity'].sum() == 3401023
    assert data['BinCount'].sum() == 366390
    # With numpy 2 (numpy 1.26: 2235.266187): the samples' angles, stored in single precision,
    # round differently, which moves one point into the next bin; the old code moves the same
    assert np.nansum(data['Int']) == pytest.approx(2235.265437, rel=1e-9)
    assert data['intensity'].dtype == aligned_points['intensity'].dtype
    assert np.allclose(data['Energy'], 0.0) # Middle of the range
    assert np.diff(result.bins['qx'][0]) == pytest.approx(0.05)

def test_cut_q_plane_energy_bins(aligned_points):
    result = Cuts.cut_q_plane(**aligned_points, energy_bins=[-0.6, -0.2, 0.2],
                              x_bin_tolerance=0.05, y_bin_tolerance=0.05)
    assert len(result.bins['qx']) == 2 # Each energy range binned on its own
    assert sorted(set(result.data['Energy'])) == pytest.approx([-0.4, 0.0])
    assert result.data['H'].isna().all() # No sample: no H, K, L
    with pytest.raises(AttributeError):
        Cuts.cut_q_plane(**aligned_points, energy_min=0.2, energy_max=-0.2)
    with pytest.raises(AttributeError):
        Cuts.cut_q_plane(**aligned_points)

# Background subtraction (checked against the old cut1D, cutQE and plotQPlane with
# backgroundSubtraction: identical for cut1D; for cutQE and plotQPlane equal to 1e-16, apart
# from the columns below, which now follow cut1D)

def test_background_columns(points, q1q2):
    common = dict(**points, q1=q1q2[0], q2=q1q2[1], width=0.1, min_pixel=0.02)
    half = 0.5*points['intensity'] # Half the counts as background
    cuts = [lambda **kwargs: Cuts.cut_1d(**common, energy_min=-1.0, energy_max=0.8, **kwargs),
            lambda **kwargs: Cuts.cut_qe(**common, energy_min=-1.0, energy_max=0.8,
                                         energy_step=0.1, **kwargs),
            lambda **kwargs: Cuts.cut_q_plane(**points, energy_min=-0.2, energy_max=0.2,
                                              **kwargs)]
    for cut in cuts:
        plain, subtracted = cut().data, cut(background=half).data
        assert list(subtracted.columns) == list(plain.columns) + [COLUMNS[key] for key in [
            'BackgroundIntensity', 'ForegroundIntensity', 'Background', 'Foreground',
            'ForegroundError', 'BackgroundError']]
        # The counts are kept; the background is summed like them
        assert np.array_equal(subtracted['intensity'], plain['intensity'])
        assert np.array_equal(subtracted['intensity_Fg'], plain['intensity'])
        assert np.allclose(subtracted['intensity_Bg'], 0.5*plain['intensity'])
        assert np.allclose(subtracted['Int_Fg'], plain['Int'], equal_nan=True)
        assert np.allclose(subtracted['Int'], 0.5*plain['Int'], equal_nan=True)
        scale = plain['BinCount']/(plain['Normalization']*plain['Monitor'])
        assert np.allclose(subtracted['Int_err'], np.sqrt(1.5*plain['intensity'])*scale,
                           equal_nan=True)
        assert np.allclose(subtracted['Int_Bg_err'], np.sqrt(0.5*plain['intensity'])*scale,
                           equal_nan=True)

def test_background_empty_cut(points, q1q2):
    empty = Cuts.cut_1d(**points, q1=q1q2[0], q2=q1q2[1], width=0.1, min_pixel=0.02,
                        energy_min=50, energy_max=60, background=points['intensity'])
    assert len(empty) == 0
    assert 'Int_Bg' in empty.data.columns

def test_cut_qe_energy_max(points, q1q2):
    # As in the old cutQE: points above energy_max are left out of the last bin, which
    # reaches beyond it
    common = dict(**points, q1=q1q2[0], q2=q1q2[1], width=0.1, min_pixel=0.02,
                  energy_min=-1.0, energy_step=0.3)
    cut = Cuts.cut_qe(**common, energy_max=0.0)
    assert cut.bins['energy'][-1] == pytest.approx(0.2)
    last = cut.data[cut.data['Energy'] == cut.bins['energy'][-2]]
    full = Cuts.cut_qe(**common, energy_max=0.2).data
    assert last['BinCount'].sum() < full[full['Energy'] == cut.bins['energy'][-2]]['BinCount'].sum()

# Voxels (bin_3d; edges and sums checked against the old binData3D: identical)

def test_grid_edges():
    edges = Cuts.grid_edges(0.25, [0.0, 0.3, 1.0])
    assert edges == pytest.approx([-0.125, 0.125, 0.375, 0.625, 0.875, 1.125])
    # The step is adjusted to divide the range evenly; the end bins are centred on the ends
    edges = Cuts.grid_edges(0.3, [0.0, 1.0])
    assert np.diff(edges) == pytest.approx([1/3]*4)
    assert 0.5*(edges[0] + edges[1]) == pytest.approx(0.0)
    with pytest.raises(AttributeError): # A single bin
        Cuts.grid_edges(1.0, [0.0, 0.2])

def test_bin_3d(dataset, aligned_points):
    from MJOLNIR.Data.Coordinates import align_to_lattice
    result = Cuts.bin_3d(**aligned_points, dx=0.1, dy=0.1, dz=0.5,
                         sample=align_to_lattice(dataset.sample))
    shape = tuple(len(result.bins[axis]) - 1 for axis in ['qx', 'qy', 'energy'])
    assert len(result) == np.prod(shape)
    assert result.grid('Int').shape == shape
    # Every point is in a voxel
    assert result.data['BinCount'].sum() == len(aligned_points['qx'])
    assert result.data['intensity'].sum() == aligned_points['intensity'].sum()
    # Rows in C order of (Qx, Qy, energy); positions are the voxel centres
    centres = 0.5*(result.bins['energy'][:-1] + result.bins['energy'][1:])
    assert np.allclose(result.grid('Energy')[0, 0], centres)
    assert np.allclose(result.grid('Qx')[:, 0, 0], 0.5*(result.bins['qx'][:-1] + result.bins['qx'][1:]))
    assert not result.data['H'].isna().any()
    scale = result.data['BinCount']/(result.data['Normalization']*result.data['Monitor'])
    assert np.allclose(result.data['Int'], result.data['intensity']*scale, equal_nan=True)
    # The same edges again
    again = Cuts.bin_3d(**aligned_points, bins=[result.bins[axis] for axis in ['qx', 'qy', 'energy']])
    assert np.array_equal(again.data['intensity'], result.data['intensity'])
    with pytest.raises(AttributeError):
        Cuts.bin_3d(**aligned_points, dx=0.1, dy=0.1)

def test_bin_3d_background(aligned_points):
    plain = Cuts.bin_3d(**aligned_points, dx=0.1, dy=0.1, dz=0.5)
    subtracted = Cuts.bin_3d(**aligned_points, dx=0.1, dy=0.1, dz=0.5,
                             background=0.5*aligned_points['intensity'])
    assert np.allclose(subtracted.data['Int'], 0.5*plain.data['Int'], equal_nan=True)
    assert np.allclose(subtracted.data['intensity_Bg'], 0.5*plain.data['intensity'])

def test_grid_needs_regular_bins(aligned_points):
    plane = Cuts.cut_q_plane(**aligned_points, energy_min=-0.2, energy_max=0.2)
    with pytest.raises(ValueError):
        plane.grid('Int')
