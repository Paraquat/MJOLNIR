import pytest

import copy
import warnings
import numpy as np
from pathlib import Path

from MJOLNIR import TasUBlibDEG as TasUBlib
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName
from MJOLNIR.Data.Mask import circleMask

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')

def orientation(sample):
    """U of UB = U B"""
    return sample.UB @ np.linalg.inv(TasUBlib.calculateBMatrix(sample.cell))

def new_cell(sample):
    cell = sample.unitCell.copy()
    cell[:2] *= 1.01
    cell[5] = 91.0
    return cell

@pytest.fixture
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

@needs_data
def test_sample_update_twice():
    """The old Sample.updateSampleParameters kept the old B: a second update took U from it"""
    sample = CameaDataReader(files[0]).sample
    U = orientation(sample)
    cell = new_cell(sample)
    once = copy.deepcopy(sample)
    once.updateSampleParameters(cell)
    sample.updateSampleParameters(sample.unitCell*[1.02, 1, 1, 1, 1, 1])
    sample.updateSampleParameters(cell)
    assert np.allclose(orientation(sample), U, atol=1e-14)
    assert np.allclose(sample.UB, once.UB, atol=1e-14)

@needs_data
def test_reader():
    reader = CameaDataReader(files[0])
    reader.set_instrument_calibration(8)
    reader.transform()
    qx, U = reader.qx.copy(), orientation(reader.sample)
    cell = new_cell(reader.sample)
    reader.update_sample_parameters(cell)
    assert np.allclose(reader.sample.unitCell, cell)
    assert np.allclose(orientation(reader.sample), U, atol=1e-14)
    assert np.array_equal(reader.qx, qx)
    h, k, l = reader.h.copy(), reader.k.copy(), reader.l.copy()
    reader.transform() # The same H, K, L as from the angles
    for converted, transformed in zip((h, k, l), (reader.h, reader.k, reader.l)):
        assert np.allclose(converted, transformed, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match='a, b, c, alpha, beta, gamma'):
        reader.update_sample_parameters(cell[:3])

@needs_data
def test_dataset_kept_for_later_files(dataset):
    original = dataset.sample.unitCell.copy()
    cell = new_cell(dataset.sample)
    dataset.update_sample_parameters(cell)
    dataset.append(files[2]) # The old check refused it: its cell is the file's
    assert all(np.allclose(sample.unitCell, cell) for sample in dataset.samples)
    assert np.allclose(dataset[2].h, dataset[2].sample.calculateQxQyToHKL(dataset[2].qx,
                                                                          dataset[2].qy)[0])
    subset = dataset[0:2]
    subset.append(files[3])
    assert np.allclose(subset[2].sample.unitCell, cell)
    dataset.set_binning(3) # The cell stays: it is on the samples
    assert all(np.allclose(sample.unitCell, cell) for sample in dataset.samples)
    other = CameaDataReader(files[4])
    other.sample.updateSampleParameters(original*[1.1, 1, 1, 1, 1, 1])
    with pytest.raises(ValueError, match='neither'):
        dataset.append(other)
    with pytest.raises(ValueError, match='derived data'):
        (dataset - dataset).update_sample_parameters(original)
    dataset.update_sample_parameters(original) # Back to the files' cell
    assert all(np.allclose(sample.unitCell, original) for sample in dataset.samples)

@needs_data
def test_updated_while_empty():
    dataset = DataSet(InstrumentName.CAMEA)
    cell = new_cell(CameaDataReader(files[0]).sample)
    dataset.update_sample_parameters(cell)
    dataset.extend(files[:2])
    assert all(np.allclose(sample.unitCell, cell) for sample in dataset.samples)

@needs_data
def test_masks_and_background_follow(dataset):
    mask = circleMask(center=[-1.0, 0.0], radius=0.2, coordinates=[ArrayType.H, ArrayType.K])
    dataset.mask = mask
    dataset.generate_background_model()
    masked = [reader.user_mask.copy() for reader in dataset]
    cell = new_cell(dataset.sample)
    dataset.update_sample_parameters(cell)
    assert not all(np.array_equal(reader.user_mask, before)
                   for reader, before in zip(dataset, masked)) # The mask moved
    expected = DataSet(InstrumentName.CAMEA, files=files[:2])
    expected.update_sample_parameters(cell)
    expected.mask = mask
    expected.generate_background_model()
    for reader, other in zip(dataset, expected):
        assert np.array_equal(reader.mask, other.mask)
        assert np.allclose(reader.background, other.background, equal_nan=True)
