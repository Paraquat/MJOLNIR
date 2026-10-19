import pytest

import dataclasses
import h5py
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.CameaDataWriter import CameaDataWriter
from MJOLNIR.Data.NexusLayout import Field, Group, NexusLayout
from MJOLNIR.Data.NexusDataWriter import NexusDataWriter

SAMPLE_DATA = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not SAMPLE_DATA, reason='sampledata/camea2024*.hdf not available')

class ToyLayout(NexusLayout):
    instrument = 'TOY'
    groups = {**NexusLayout.groups,
              'detector': Group('{entry}/{instrument}/detector', 'NXdetector')}
    fields = {'A4': Field('{entry}/{instrument}/a4', units='degree', dtype='float32'),
              'counts': Field('{entry}/{instrument}/detector/data', units='counts',
                              dtype='int32')}

@pytest.fixture
def toy(tmp_path):
    source = tmp_path / 'toy.hdf'
    with h5py.File(source, 'w') as f:
        f['entry/TOY/a4'] = np.array([1.0, 2.0], dtype='float32')
        f['entry/TOY/a4'].attrs['units'] = b'degree'
        f['entry/TOY/a4'].attrs.create('note', b'kept', dtype='S4') # Fixed length, as NICOS
        f['entry/TOY/a4_link'] = f['entry/TOY/a4'] # Hard link, as NICOS's entry/data/*
        f['entry/other'] = [7]
    return SimpleNamespace(file=source, layout=ToyLayout)

def test_nexus_writer_updates_a_copy(toy, tmp_path):
    writer = NexusDataWriter(toy)
    writer.set('A4', [3.0, 4.0])
    target = writer.save(tmp_path / 'out.hdf')
    with h5py.File(target) as f:
        assert np.array_equal(f['entry/TOY/a4'][()], [3.0, 4.0])
        assert f['entry/TOY/a4'].attrs['note'] == b'kept'
        assert np.array_equal(f['entry/TOY/a4_link'][()], [3.0, 4.0]) # Written in place
        assert f['entry/other'][0] == 7 # Not in the layout, kept
    with h5py.File(toy.file) as f:
        assert np.array_equal(f['entry/TOY/a4'][()], [1.0, 2.0]) # Source untouched

def test_nexus_writer_replaces_and_creates(toy, tmp_path):
    writer = NexusDataWriter(toy)
    writer.set('A4', [1.0, 2.0, 3.0]) # New shape: dataset replaced, attributes kept
    writer.set('counts', np.arange(6).reshape(2, 3)) # New dataset and group
    with h5py.File(writer.save(tmp_path / 'out.hdf')) as f:
        assert f['entry/TOY/a4'].shape == (3,)
        assert f['entry/TOY/a4'].dtype == np.float32
        assert f['entry/TOY/a4'].attrs['note'] == b'kept'
        assert f['entry/TOY/a4'].attrs.get_id('note').dtype == np.dtype('S4') # Type kept
        assert f['entry/TOY/a4'].attrs['units'] == 'degree' # Already right: kept as stored
        assert f['entry/TOY/detector'].attrs['NX_class'] == b'NXdetector'
        assert f['entry/TOY/detector/data'].dtype == np.int32
        assert f['entry/TOY/detector/data'].attrs['units'] == b'counts' # Missing: written
    with pytest.raises(KeyError):
        writer.set('notAField', 1)

def test_nexus_writer_targets(toy, tmp_path):
    writer = NexusDataWriter(toy)
    with pytest.raises(ValueError, match='read from'):
        writer.save(toy.file)
    with pytest.raises(ValueError, match='read from'):
        writer.save(tmp_path / '.' / 'toy.hdf')
    (tmp_path / 'exists.hdf').write_bytes(b'')
    with pytest.raises(FileExistsError):
        writer.save(tmp_path / 'exists.hdf')
    writer.save(tmp_path / 'exists.hdf', overwrite=True)
    assert h5py.is_hdf5(tmp_path / 'exists.hdf')

def test_failed_save_leaves_no_file(toy, tmp_path):
    writer = NexusDataWriter(toy)
    writer.values['A4'] = np.array(['not', 'numbers']) # Fails when written
    with pytest.raises(Exception):
        writer.save(tmp_path / 'out.hdf')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['toy.hdf']

def test_writer_layout_must_match_reader(toy):
    with pytest.raises(ValueError, match='writes CameaLayout files'):
        CameaDataWriter(SimpleNamespace(file=toy.file, layout=ToyLayout, calibration=None))

def paths(f):
    names = []
    f.visit(names.append)
    return sorted(names)

def snapshot(reader):
    return {'counts': reader.counts, 'A3': reader.A3, 'A4': reader.A4, 'Ei': reader.Ei,
            'monitor1': reader.monitor1, 'monitor2': reader.monitor2,
            'environment': dataclasses.asdict(reader.environment),
            'metadata': dict(reader._metadata), 'UB': reader.sample.orientationMatrix,
            'binnings': reader.possible_binnings}

def assert_same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif a is None or isinstance(a, (str, list)):
        assert a == b
    else:
        assert np.array_equal(a, b, equal_nan=True)

@needs_data
def test_camea_writer_new_calibration(tmp_path):
    reader = CameaDataReader(SAMPLE_DATA[0])
    reader.set_instrument_calibration(3)
    calibration = dataclasses.replace(reader.calibration, Ef=reader.calibration.Ef + 0.5)
    target = CameaDataWriter(reader, {3: calibration}).save(tmp_path / 'recalibrated.hdf')

    saved = CameaDataReader(target)
    assert np.array_equal(saved.get_instrument_calibration(3).Ef,
                          (reader.calibration.Ef + 0.5).astype(np.float32))
    for binning in (1, 5, 8):
        assert_same(dataclasses.asdict(saved.get_instrument_calibration(binning)),
                    dataclasses.asdict(reader.get_instrument_calibration(binning)))
    assert_same(snapshot(saved), snapshot(reader))
    with h5py.File(SAMPLE_DATA[0]) as source, h5py.File(target) as f:
        assert paths(f) == paths(source) # Nothing dropped or added
        assert f['entry/data/data'] == f['entry/CAMEA/detector/data'] # Hard link kept

@needs_data
def test_camea_writer_default_and_new_binning(tmp_path):
    reader = CameaDataReader(SAMPLE_DATA[0])
    reader.set_instrument_calibration(1)
    writer = CameaDataWriter(reader) # The reader's current calibration
    assert list(writer.calibrations) == [1]
    calibration = reader.get_instrument_calibration(1)
    doubled = dataclasses.replace(calibration, **{
        key: np.repeat(getattr(calibration, key), 2, axis=0) for key in ('Ef', 'width', 'bg',
                                                                         'amp', 'A4', 'bound')})
    writer.set_calibration(2, doubled)
    saved = CameaDataReader(writer.save(tmp_path / 'binning2.hdf'))
    assert saved.possible_binnings == [1, 2, 3, 5, 8]
    assert saved.get_instrument_calibration(2).bound.shape == (1664, 2)
    with h5py.File(saved.file) as f:
        assert f['entry/CAMEA/calib2'].attrs['NX_class'] == b'NXcollection'

@needs_data
def test_camea_writer_checks_calibration_shape(tmp_path):
    reader = CameaDataReader(SAMPLE_DATA[0])
    calibration = reader.get_instrument_calibration(1)
    with pytest.raises(ValueError, match='expected'):
        CameaDataWriter(reader, {3: calibration}).save(tmp_path / 'wrong.hdf')
    assert not (tmp_path / 'wrong.hdf').exists()
