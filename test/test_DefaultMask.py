import pytest

import warnings
import numpy as np

from MJOLNIR.Data import Mask
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.MarmotDataReader import MarmotDataReader
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName


def camea_reader(camea_2024_files, binning):
    reader = CameaDataReader(camea_2024_files[0])
    reader.set_instrument_calibration(binning)
    reader.transform()
    return reader

def test_camea_binning_8(camea_2024_files):
    # As the old DataFile conversion: the first two energy bins at binning 8
    reader = camea_reader(camea_2024_files, 8)
    assert reader.mask.sum() == 135 * 104 * 2
    assert reader.mask[:, :, :2].all()
    assert not reader.mask[:, :, 2:].any()
    assert not reader.user_mask.any()
    assert not camea_reader(camea_2024_files, 3).mask.any()

def test_nan_calibration(camea_2024_files):
    # Pixels without a calibration (NaN A4, Ef or amplitude) cannot be used, on any instrument
    reader = camea_reader(camea_2024_files, 3)
    reader.calibration.A4.reshape(104, 24)[5] = np.nan # A broken detector tube
    reader.calibration.Ef.reshape(104, 24)[7, 3] = np.nan # One pixel
    with pytest.warns(UserWarning, match='Detectors 5 of'):
        reader.transform()
    assert reader.masked_detectors == [5]
    assert reader.mask[:, 5].all()
    assert reader.mask[:, 7, 3].all()
    assert reader.mask.sum() == 135 * 24 + 135
    # The same detectors are reported only once
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        reader.transform()

def test_marmot_low_Ef(marmot_file, tmp_path):
    # As the old loadMarmot: pixels with calibrated Ef below 1 meV
    rows = [[det, pix, 0, 1.0, 0.5 if pix < 10 else 2.0, 0.1, 0.0, pix, pix + 1, 2.0*det]
            for det in range(30) for pix in range(256)]
    table = tmp_path / 'calibration.csv'
    np.savetxt(table, rows, delimiter=',', header='Detector,Energy,Pixel,Amplitude,Center,Width,Background,lowerBin,upperBin,A4Offset')
    reader = MarmotDataReader(marmot_file, calibration_file=table)
    reader.set_instrument_calibration()
    reader.transform()
    assert reader.mask.sum() == 361 * 30 * 10
    assert reader.mask[:, :, :10].all()

def test_user_and_default_mask(camea_2024_files):
    reader = camea_reader(camea_2024_files, 8)
    default = reader.mask.copy()
    user = np.zeros_like(default)
    user[:, 0] = True
    reader.mask = user # Sets the user's mask
    assert np.array_equal(reader.user_mask, user)
    assert np.array_equal(reader.mask, user | default)
    reader.use_default_mask = False
    assert np.array_equal(reader.mask, user)

def test_dataset_layers(camea_2024_files):
    dataset = DataSet(InstrumentName.CAMEA, files=camea_2024_files[:2]) # Binning 8
    default = [m.copy() for m in dataset.mask]
    assert all(m.sum() == 135 * 104 * 2 for m in default)
    circle = Mask.circleMask(center=[-1, 0], radius=0.25, coordinates=['h', 'k'])
    dataset.mask = circle
    assert all(np.array_equal(m, c | d)
               for m, c, d in zip(dataset.mask, circle(dataset), default))
    assert dataset.get_view(ArrayType.MASK)[0, 0, 0] == 1.0 # The view shows both
    # remove_masking removes the user's mask only
    dataset.remove_masking()
    assert all(np.array_equal(m, d) for m, d in zip(dataset.mask, default))
    # The default mask can be switched off, also for files added later
    dataset.use_default_mask = False
    dataset.append(camea_2024_files[2])
    assert not any(m.any() for m in dataset.mask)
    dataset.use_default_mask = True
    assert all(m.sum() == 135 * 104 * 2 for m in dataset.mask)

def test_dataset_binning(camea_2024_files):
    dataset = DataSet(InstrumentName.CAMEA, files=camea_2024_files[:2])
    dataset.set_binning(3) # No default mask at binning 3
    assert not any(m.any() for m in dataset.mask)
    dataset.set_binning(8)
    assert all(m.sum() == 135 * 104 * 2 for m in dataset.mask)

def test_append_mask_keeps_layers(camea_2024_files):
    # append_mask adds to the user's mask: the default mask is not copied into it
    dataset = DataSet(InstrumentName.CAMEA, files=camea_2024_files[:1])
    extra = np.zeros(dataset[0].intensity.shape, dtype=bool)
    extra[:, 1] = True
    dataset.append_mask(extra)
    assert np.array_equal(dataset[0].user_mask, extra)
    dataset.use_default_mask = False
    assert np.array_equal(dataset.mask[0], extra)
