import pytest

import shutil
import warnings
import h5py
import numpy as np

from MJOLNIR.Data.CameaDataReader import CameaDataReader, interval_means


def without(tmp_path, source, *paths):
    """Copy of a sample file with some datasets removed (no real NICOS file without a
    stored temperature is available, question 41)"""
    path = tmp_path / source.name
    shutil.copy(source, path)
    with h5py.File(path, 'r+') as f:
        for name in paths:
            del f[name]
    return path

def read_quietly(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return CameaDataReader(path)

def test_interval_means():
    times = np.array([0.0, 10.0, 20.0])
    values = np.array([0.0, 10.0, 0.0])
    means = interval_means(times, values, [0.0, 5.0, 15.0, -5.0, 30.0, 7.0],
                           [10.0, 15.0, 25.0, 0.0, 40.0, 7.0])
    # Linear between points, held before the first and after the last; an empty interval
    # gets the value at its end
    assert np.allclose(means, [5.0, 7.5, 1.25, 0.0, 0.0, 7.0])

def test_interval_means_between_log_points():
    # A step with no log point inside it gets the interpolated value
    means = interval_means(np.array([0.0, 10.0]), np.array([4.0, 6.0]), [2.0], [4.0])
    assert np.allclose(means, 4.6)

def test_stored_temperature_kept(camea_2024_files):
    reader = CameaDataReader(camea_2024_files[0])
    assert reader._metadata['temperature_from_log'] is False

def test_temperature_from_log(tmp_path, camea_2024_file): # Parametrized over the files in conftest
    source = camea_2024_file
    stored = CameaDataReader(source).environment.temperature
    reader = read_quietly(without(tmp_path, source, 'entry/sample/temperature'))
    assert reader._metadata['temperature_from_log'] is True
    assert reader.environment.units['temperature'] == 'K'
    temperature = reader.environment.temperature
    assert temperature.shape == stored.shape
    # As NICOS's own value per step, within the scatter of the log
    assert np.sqrt(np.mean((temperature - stored)**2)) < 0.005
    assert np.sqrt(np.mean((temperature - stored)**2)) < 0.5*np.std(stored)

@pytest.mark.parametrize('missing', ['entry/sample/temperature_log',
                                     'entry/control/absolute_time'])
def test_no_log(camea_2024_files, tmp_path, missing):
    reader = read_quietly(without(tmp_path, camea_2024_files[0], 'entry/sample/temperature', missing))
    assert reader.environment.temperature is None
    assert reader._metadata['temperature_from_log'] is False

def test_log_without_start(camea_2024_files, tmp_path):
    path = without(tmp_path, camea_2024_files[0], 'entry/sample/temperature')
    with h5py.File(path, 'r+') as f:
        del f['entry/sample/temperature_log/time'].attrs['start']
    with pytest.warns(UserWarning, match='no start time'):
        reader = CameaDataReader(path)
    assert reader.environment.temperature is None
