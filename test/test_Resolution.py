import pytest

import warnings
import numpy as np
from pathlib import Path

from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName
from MJOLNIR.Data.Resolution import camea_resolution_matrix, project

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')

@pytest.fixture
def dataset():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return DataSet(InstrumentName.CAMEA, files=files[:1])

def test_matrix():
    M = camea_resolution_matrix(1.0, 0.5, Ei=7.0, Ef=4.0, analyser=3)
    assert M.shape == (4, 4) and np.allclose(M, M.T, rtol=1e-12, atol=1e-9*np.abs(M).max())
    assert np.all(np.linalg.eigvalsh(M[[0, 1, 3]][:, [0, 1, 3]]) > 0) # An ellipsoid in Qx, Qy, E
    # The analyser sets the sample-analyser distance, so the resolution
    assert not np.allclose(M, camea_resolution_matrix(1.0, 0.5, 7.0, 4.0, analyser=7))
    with pytest.raises(ValueError, match='analysers 0 to 7'):
        camea_resolution_matrix(1.0, 0.5, 7.0, 4.0, analyser=8)

def test_project():
    M = np.diag([4.0, 9.0, 1.0, 16.0]) # Uncorrelated: widths 1/sqrt of the diagonal
    axes, sigma = project(M, [1, 0, 0, 0], [0, 0, 0, 1])
    assert np.allclose(sorted(sigma), [0.25, 0.5])
    assert np.allclose(np.abs(axes), np.eye(2))

@needs_data
def test_independent_of_binning(dataset):
    """The old code took the analyser as the Ef bin index // 8, right only at binning 8"""
    M = dataset.resolution_matrix([-1.0, 0.0, 0.0], 4.0, 3.5)
    for binning in (1, 3):
        dataset.set_binning(binning)
        assert np.array_equal(dataset.resolution_matrix([-1.0, 0.0, 0.0], 4.0, 3.5), M)

@needs_data
def test_analyser_from_ef(dataset):
    """The analyser whose calibrated Ef is nearest: each of CAMEA's eight in turn"""
    reader = dataset[0]
    Ef = np.nanmean(reader.calibration.Ef.reshape(reader.n_detectors, reader.n_analysers,
                                                  reader.binning), axis=(0, 2))
    matrices = [dataset.resolution_matrix([0.5, 0.5, 0.0], 8.0, value) for value in Ef]
    qx, qy = dataset.convert_to_qxqy([0.5, 0.5, 0.0])
    for analyser, (value, M) in enumerate(zip(Ef, matrices)):
        assert np.array_equal(M, camea_resolution_matrix(qx, qy, 8.0, value, analyser,
                                                         a3_offset=dataset.sample.theta))

def test_other_instruments():
    with pytest.raises(NotImplementedError, match='only implemented for CAMEA'):
        DataSet(InstrumentName.BAMBUS).resolution_matrix([1.0, 0.0, 0.0], 5.0, 5.0)
    with pytest.raises(ValueError, match='no data files'):
        DataSet(InstrumentName.CAMEA).resolution_matrix([1.0, 0.0, 0.0], 5.0, 5.0)

@needs_data
def test_find_ei_and_second_order(dataset):
    Ei = float(np.unique(np.round(dataset[0].Ei, 2))[0])
    assert np.array_equal(dataset.find_ei(0.0), [Ei]) # Elastic is measured
    assert dataset.find_ei(Ei).size == 0 # Ef = 0 is not
    assert np.isnan(dataset.second_order_elastic([0.0])).all() # 3/4 Ei is outside the data
    # |4 (Ei - E) - Ei| = 8.4 at E = 0.9 (inside the data): matched with a wide tolerance
    result = dataset.second_order_elastic([0.9, 0.75*Ei], atol=9.0)
    assert np.isclose(result[0], dataset[0].Ei.mean()) and np.isnan(result[1])
    assert np.isnan(dataset.second_order_elastic([0.9], atol=8.0)).all()
