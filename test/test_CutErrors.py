import pytest

import warnings
import numpy as np
from pathlib import Path

from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')
HKL1, HKL2 = [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
ENERGY_BINS = np.linspace(-0.5, 1.5, 6)

# Each cut of the DataSet that takes measurement errors
CUTS = {
    'cut_1d': lambda ds, **o: ds.cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8, **o),
    'cut_constant_q': lambda ds, **o: ds.cut_constant_q(HKL1, **o),
    'cut_qe': lambda ds, **o: ds.cut_qe(HKL1, HKL2, 0.1, 0.05, energy_bins=ENERGY_BINS, **o),
    'cut_qe_line': lambda ds, **o: ds.cut_qe_line([HKL1, HKL2, [0.5, 0.5, 0.0]], ENERGY_BINS, **o),
    'cut_powder': lambda ds, **o: ds.cut_powder(np.linspace(0.5, 2.0, 16), ENERGY_BINS, **o),
    'cut_q_plane': lambda ds, **o: ds.cut_q_plane(energy_min=0.0, energy_max=0.5, **o),
    'bin_3d': lambda ds, **o: ds.bin_3d(0.05, 0.05, 0.2, **o),
}

@pytest.fixture(scope='module')
def datasets():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        first = DataSet(InstrumentName.CAMEA, files=files[:2])
        second = DataSet(InstrumentName.CAMEA, files=files[2:4])
    return first, second

def data(dataset, cut, **options):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return CUTS[cut](dataset, **options).data

def set_errors(dataset, factor):
    for reader in dataset:
        reader.intensity_error = None if factor is None else factor*np.sqrt(reader.intensity)

@needs_data
@pytest.mark.parametrize('cut', CUTS)
def test_errors_used(datasets, cut):
    dataset = datasets[0]
    plain = data(dataset, cut)
    try:
        set_errors(dataset, 1.0) # The counting errors given explicitly: the same result
        same = data(dataset, cut)
        set_errors(dataset, 2.0)
        doubled = data(dataset, cut)
    finally:
        set_errors(dataset, None)
    assert np.allclose(same['Int_err'], plain['Int_err'], rtol=1e-12, equal_nan=True)
    assert np.allclose(doubled['Int_err'], 2*plain['Int_err'], rtol=1e-12, equal_nan=True)
    assert np.array_equal(doubled['Int'], plain['Int'], equal_nan=True)

@needs_data
@pytest.mark.parametrize('cut', CUTS)
def test_errors_of_a_difference(datasets, cut):
    """Counts minus scaled counts can be negative: their root gave NaN errors"""
    first, second = datasets
    difference = data(first - second, cut)
    filled = difference['BinCount'] > 0
    assert filled.any()
    assert np.isfinite(difference['Int_err'][filled]).all()
    assert (difference['Int_err'][filled] >= 0).all() # 0 where both sets counted nothing
    assert (difference['Int'][filled] < 0).any() # Bins the old errors made NaN

@needs_data
@pytest.mark.parametrize('cut', ['cut_1d', 'cut_qe', 'cut_q_plane', 'bin_3d'])
def test_errors_with_background(datasets, cut):
    dataset = datasets[0]
    dataset.generate_background_model()
    try:
        plain = data(dataset, cut, background_subtraction=True)
        set_errors(dataset, 2.0)
        doubled = data(dataset, cut, background_subtraction=True)
    finally:
        set_errors(dataset, None)
        dataset.remove_background_model()
    scale = plain['BinCount']/(plain['Normalization']*plain['Monitor'])
    expected = np.sqrt(4*plain['intensity'] + plain['intensity_Bg'])*scale
    assert np.allclose(doubled['Int_err'], expected, rtol=1e-12, equal_nan=True)
    assert np.allclose(doubled['Int_Fg_err'], 2*plain['Int_Fg_err'], rtol=1e-12, equal_nan=True)
    assert np.allclose(doubled['Int_Bg_err'], plain['Int_Bg_err'], rtol=1e-12, equal_nan=True)
