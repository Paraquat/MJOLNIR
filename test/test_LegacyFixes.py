import pytest

import warnings
import numpy as np
from pathlib import Path

from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.LegacyFixes import CameaMonitor2018
from MJOLNIR.Data.NexusDataWriter import NexusDataWriter

SAMPLE_DATA = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
pytestmark = pytest.mark.skipif(not SAMPLE_DATA, reason='sampledata/camea2024*.hdf not available')

def copy_with(tmp_path, name, **values):
    """Copy of a sample file with some layout fields replaced (no real 2018 file is
    available, question 42)"""
    reader = CameaDataReader(SAMPLE_DATA[0])
    writer = NexusDataWriter(reader)
    for key, value in values.items():
        writer.set(key, value)
    return reader, writer.save(tmp_path / f'{name}.hdf')

def read_quietly(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return CameaDataReader(path)

def wrong_monitor(reader):
    """The normalisation monitor (monitor2 in the NICOS sample file), halved"""
    return reader.monitor2 // 2

def test_current_file_is_unchanged():
    reader = CameaDataReader(SAMPLE_DATA[0])
    assert reader._metadata['legacy_fixes'] == []
    assert CameaDataReader.legacy_fixes[0].applies(reader) is False

def test_2018_monitor_replaced_by_preset(tmp_path):
    original, _ = copy_with(tmp_path, 'unused')
    _, path = copy_with(tmp_path, 'monitor2018', startTime=[b'2018-03-01 10:00:00'],
                        monitor2=wrong_monitor(original))
    with pytest.warns(UserWarning, match='2018 monitor replaced by the preset'):
        reader = CameaDataReader(path)
    assert reader.scan_monitor_key() == 'monitor2'
    assert np.all(reader.monitor2 == 62500)
    assert np.all(reader.scan_monitor() == reader._metadata['preset'])
    assert np.array_equal(reader.monitor1, original.monitor1) # Not used for normalisation
    assert reader._metadata['legacy_fixes'] == [CameaMonitor2018.description]
    stored = CameaDataReader.shallow_read(path, ['monitor2'])['monitor2']
    assert np.array_equal(stored, wrong_monitor(original)) # As stored

@pytest.mark.parametrize('name, values', [
    ('other_year', {'startTime': [b'2019-03-01 10:00:00']}),
    ('time_mode', {'startTime': [b'2018-03-01 10:00:00'], 'mode': [b'time']}),
])
def test_not_applied(tmp_path, name, values):
    original, _ = copy_with(tmp_path, 'unused')
    _, path = copy_with(tmp_path, name, monitor2=wrong_monitor(original), **values)
    reader = read_quietly(path)
    assert reader._metadata['legacy_fixes'] == []
    assert np.array_equal(reader.monitor2, wrong_monitor(original))

def test_right_monitor_not_replaced(tmp_path):
    # The sample file's monitor2 already equals the preset
    _, path = copy_with(tmp_path, 'right2018', startTime=[b'2018-03-01 10:00:00'])
    reader = read_quietly(path)
    assert reader._metadata['legacy_fixes'] == []
