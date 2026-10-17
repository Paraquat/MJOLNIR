import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data import Mask
from MJOLNIR.Data.Mask import calculateCurratAxeMask
from MJOLNIR.Data.BambusDataReader import BAMBUSDataReader
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

data_path = Path(__file__).parent.parent / 'sampledata'
files = sorted(data_path.glob('camea2024*.hdf'))
bambus_file = Path.home() / 'projects' / 'CoRhO4' / 'corh2o4_test.dat'

bragg = [[1, 0, 0], [0, 1, 0]]

@pytest.fixture(scope="module")
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2]) # Default binning 8

# Masked points in camea2024n001486 at binning 8, identical to the old
# DataFile.calculateCurratAxeMask when it was moved here
@pytest.mark.parametrize('radii, spurionType, masked', [
    (dict(dqx=0.05, dqy=0.08), 'both', 4410),
    (dict(dqx=0.05, dqy=0.08), 'monochromator', 3290),
    (dict(dqx=0.05, dqy=0.08), 'analyser', 1648),
    (dict(dH=0.05, dK=0.05, dL=0.1), 'both', 3419),
    (dict(dH=0.05, dK=0.05, dL=0.1), 'monochromator', 2768),
    (dict(dH=0.05, dK=0.05, dL=0.1), 'analyser', 981),
])
def test_masked_points(dataset, radii, spurionType, masked):
    reader = dataset[0]
    inside = calculateCurratAxeMask(reader, bragg, spurionType=spurionType, **radii)
    assert inside.shape == reader.intensity.shape
    assert inside.sum() == masked
    outside = calculateCurratAxeMask(reader, bragg, spurionType=spurionType, maskInside=False,
                                     **radii)
    assert np.array_equal(outside, ~inside)

def test_both_is_union(dataset):
    reader = dataset[0]
    radii = dict(dqx=0.05, dqy=0.08)
    both = calculateCurratAxeMask(reader, bragg, spurionType='both', **radii)
    mono = calculateCurratAxeMask(reader, bragg, spurionType='monochromator', **radii)
    analyser = calculateCurratAxeMask(reader, bragg, spurionType='analyser', **radii)
    assert np.array_equal(both, mono | analyser)

def test_mask_object_on_dataset(dataset):
    mask = Mask.CurratAxeMask(bragg, dqx=0.05, dqy=0.08)
    result = mask(dataset)
    assert len(result) == 2
    assert np.array_equal(result[0], calculateCurratAxeMask(dataset[0], bragg, dqx=0.05, dqy=0.08))
    assert result[0].sum() == 4410

def test_other_instrument():
    # Not specific to CAMEA: any reader with an Ei, calibrated Ef and sample works
    reader = BAMBUSDataReader(bambus_file)
    reader.set_instrument_calibration()
    reader.transform()
    mask = Mask.CurratAxeMask([[1, 1, 1], [2, 2, 0], [0, 0, 4]], dqx=0.1, dqy=0.1)(reader)
    assert mask.shape == (101, 20, 5)
    assert mask.sum() == 119

def test_errors(dataset):
    with pytest.raises(AttributeError): # Neither dqx/dqy nor dH/dK/dL
        calculateCurratAxeMask(dataset[0], bragg, dqx=0.05)
    with pytest.raises(AttributeError):
        calculateCurratAxeMask(dataset[0], bragg, dqx=0.05, dqy=0.05, spurionType='sample')
    with pytest.raises(AttributeError): # Only data files can be masked
        Mask.CurratAxeMask(bragg, dqx=0.05, dqy=0.05)(np.linspace(0, 1, 11))
