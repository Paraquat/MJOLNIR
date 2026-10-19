import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.Export import to_ufit
from MJOLNIR.Data.InstrumentName import InstrumentName

ufit = pytest.importorskip('ufit')
files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
pytestmark = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')
HKL1, HKL2 = [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]

@pytest.fixture(scope="module")
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

def test_along_q(dataset):
    result = dataset.cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8)
    data = dataset.generate_ufit_dataset(result)
    assert np.allclose(data.y, result.data['Int'], equal_nan=True)
    assert np.allclose(data.dy, result.data['Int_err'], equal_nan=True)
    assert data.x[0] == 0 and np.all(np.diff(data.x) > 0) # From the first bin, increasing
    # x in units of the direction (here H, K, 0 -> (1, 1, 0)): x * (1, 1, 0) + first bin
    hkl = result.data[['H', 'K', 'L']].to_numpy()
    assert np.allclose(hkl[0] + np.outer(data.x, [1, 1, 0]), hkl, atol=1e-6)
    assert data.xcol == 'H-1.03\nK-0.03\n0\n-0.1\n[RLU,meV]' # As the old generateUFitDataset
    assert data.meta['instrument'] == 'CAMEA'
    assert data.meta['datafilename'] == ', '.join(f.name for f in files[:2])

def test_along_q_without_rlu(dataset):
    q1, q2 = dataset.convert_to_qxqy([HKL1, HKL2])
    result = dataset.cut_1d(q1, q2, 0.1, 0.02, -1.0, 0.8, rlu=False)
    data = to_ufit(result, rlu=False)
    assert 'Qx' in data.xcol and data.xcol.endswith(r'[$\AA^{-1}$,meV]')
    assert np.all(np.diff(data.x) > 0)

def test_along_energy(dataset):
    result = dataset.cut_constant_q([0.0, 0.2, 0.0], width=0.1, min_pixel=0.05,
                                    energy_min=-0.8, energy_max=0.6)
    data = dataset.generate_ufit_dataset(result)
    assert np.allclose(data.x, result.data['Energy'])
    assert np.allclose(data.dy, result.data['Int_err'], equal_nan=True)
    assert data.xcol == 'E [meV] at (0.0, 0.2, 0.0) [RLU]'

def test_errors_with_background(dataset):
    dataset.generate_background_model()
    try:
        result = dataset.cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8, background_subtraction=True)
        data = to_ufit(result)
        # Int_err includes the background (the old export used the counts only)
        assert np.allclose(data.dy, result.data['Int_err'], equal_nan=True)
        assert np.allclose(data.y, result.data['Int'], equal_nan=True)
    finally:
        dataset.remove_background_model()

def test_refuses(dataset):
    with pytest.raises(ValueError, match='1D cuts'):
        to_ufit(dataset.cut_qe(HKL1, HKL2, 0.1, 0.02, energy_min=-1.0, energy_max=0.8,
                               energy_step=0.2))
    q1, q2 = dataset.convert_to_qxqy([HKL1, HKL2])
    from MJOLNIR.Data import Cuts
    points = dataset._points(rotate=False)
    no_sample = Cuts.cut_1d(**points, q1=q1, q2=q2, width=0.1, min_pixel=0.02,
                            energy_min=-1.0, energy_max=0.8)
    with pytest.raises(AttributeError, match='rlu=False'):
        to_ufit(no_sample)
    empty = dataset.cut_constant_q([50.0, 50.0, 0.0], energy_min=-1.0, energy_max=0.8)
    with pytest.raises(ValueError, match='empty'):
        to_ufit(empty)
