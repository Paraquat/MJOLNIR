import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.NexusDataWriter import NexusDataWriter

SAMPLE_DATA = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
pytestmark = pytest.mark.skipif(not SAMPLE_DATA, reason='sampledata/camea2024*.hdf not available')

def running_scan(tmp_path, extra, keys=('monitor1', 'monitor2', 'time')):
    """Copy of a sample file with `extra` values appended to the given per-step arrays, as
    NICOS leaves them in a file opened during a scan"""
    reader = CameaDataReader(SAMPLE_DATA[0])
    writer = NexusDataWriter(reader)
    for key in keys:
        values = np.asarray(getattr(reader, key))
        writer.set(key, np.concatenate([values, values[-1:].repeat(extra)]))
    return reader, writer.save(tmp_path / f'running{extra}.hdf')

def test_finished_scan_is_unchanged():
    reader = CameaDataReader(SAMPLE_DATA[0])
    assert reader._metadata['unfinished'] is False
    assert reader.monitor1.shape == reader.monitor2.shape == reader.time.shape == (reader.n_steps,)

def test_extra_monitor_value_is_dropped(tmp_path):
    finished, path = running_scan(tmp_path, 1)
    with pytest.warns(UserWarning, match='scan is unfinished'):
        reader = CameaDataReader(path)
    for key in ('monitor1', 'monitor2', 'time'):
        assert np.array_equal(getattr(reader, key), getattr(finished, key)), key
    assert reader._metadata['unfinished'] is True
    reader.set_instrument_calibration(8)
    reader.transform() # The step-count check passes
    assert reader.monitor.shape[0] == reader.n_steps

def test_only_the_normalisation_monitor_too_long(tmp_path):
    finished, path = running_scan(tmp_path, 1, keys=('monitor2',))
    with pytest.warns(UserWarning, match='scan is unfinished'):
        reader = CameaDataReader(path)
    assert np.array_equal(reader.monitor2, finished.monitor2)
    assert np.array_equal(reader.monitor1, finished.monitor1) # Already the right length

def test_two_extra_values_are_refused(tmp_path):
    _, path = running_scan(tmp_path, 2)
    reader = CameaDataReader(path) # Not a running scan: left as stored
    assert reader._metadata['unfinished'] is False
    reader.set_instrument_calibration(8)
    with pytest.raises(ValueError, match='monitor values for'):
        reader.transform()
