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
