import pytest

import h5py
import numpy as np

from MJOLNIR.Data.BambusDataReader import BAMBUSDataReader
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.DataReaderFactory import DataReaderFactory
from MJOLNIR.Data.InstrumentName import InstrumentName


@pytest.fixture
def camea_like(tmp_path):
    """The few datasets of a CAMEA file that shallow_read needs: no sample, no calibration"""
    path = tmp_path / 'camea_like.hdf'
    with h5py.File(path, 'w') as f:
        f['entry/CAMEA/detector/data'] = np.zeros((2, 3, 4), dtype='uint32')
        f['entry/CAMEA/analyzer/polar_angle'] = np.array([-44.0], dtype='float32')
        f['entry/CAMEA/analyzer/polar_angle_offset'] = np.array([-4.0], dtype='float32')
        f['entry/title'] = np.array([b'A title'])
        f['entry/sample/T'] = [1.5, 2.5] # The 2023 name of the temperature
        f['entry/scanvars'] = np.array([b'a3'])
    return path

def test_shallow_read_nexus(camea_like):
    values = CameaDataReader.shallow_read(camea_like, ['name', 'fileLocation', 'twoTheta',
                                                       'title', 'temperature', 'scanVariables',
                                                       'Ei', 'comment'])
    assert values['name'] == 'camea_like.hdf'
    assert values['fileLocation'] == str(camea_like.parent)
    assert values['twoTheta'] == pytest.approx([-40.0])
    assert values['title'] == 'A title' # Metadata decoded
    assert values['scanVariables'] == 'a3' # A single string decoded
    assert np.array_equal(values['temperature'], [1.5, 2.5]) # Found under its alias
    assert values['Ei'] is None and values['comment'] is None # Not in the file

def test_shallow_read_errors(camea_like, tmp_path):
    with pytest.raises(KeyError, match='notAKey'):
        CameaDataReader.shallow_read(camea_like, ['title', 'notAKey'])
    text_file = tmp_path / 'text.dat'
    text_file.write_text('not hdf')
    with pytest.raises(ValueError, match='not a CAMEA data file'):
        CameaDataReader.shallow_read(text_file, ['title'])
    with pytest.raises(KeyError):
        DataReaderFactory().shallow_read(InstrumentName.CAMEA, [], ['notAKey'])

def test_shallow_keys():
    keys = CameaDataReader.shallow_keys()
    assert {'name', 'fileLocation', 'twoTheta', 'A3', 'title', 'temperature',
            'protonBeam', 'singleDetector1'} <= set(keys)
    assert keys == sorted(keys, key=str.lower)
    assert 'protonBeam' not in BAMBUSDataReader.shallow_keys()
    assert {'A3', 'title', 'startTime', 'temperature'} <= set(BAMBUSDataReader.shallow_keys())

def test_shallow_read_matches_camea_reader(camea_2024_files):
    keys = ['twoTheta', 'A3', 'A4', 'Ei', *CameaDataReader.metadata_keys()]
    results = DataReaderFactory().shallow_read(InstrumentName.CAMEA, camea_2024_files[:2], keys)
    assert len(results) == 2
    for data_file, values in zip(camea_2024_files, results):
        reader = CameaDataReader(data_file)
        for key in ['A3', 'A4', 'Ei']:
            assert np.array_equal(values[key], getattr(reader, key))
        assert np.array_equal(values['twoTheta'], reader.twotheta)
        for key in CameaDataReader.metadata_keys():
            assert values[key] == reader._metadata.get(key), key

def test_shallow_read_nicos(bambus_file):
    values = BAMBUSDataReader.shallow_read(bambus_file, ['name', 'twoTheta', 'title',
                                                         'monitor1', 'temperature'])
    reader = BAMBUSDataReader(bambus_file)
    assert values['name'] == bambus_file.name
    assert np.array_equal(values['twoTheta'], reader.twotheta)
    assert values['title'] == reader._metadata['title']
    assert np.array_equal(values['monitor1'], reader.monitor1)
    assert values['temperature'] is None # Not in this file
