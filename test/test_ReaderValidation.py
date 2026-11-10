import pytest

import h5py
import shutil
import numpy as np
from pathlib import Path

from MJOLNIR.Data.DataReaderFactory import DataReaderFactory
from MJOLNIR.Data.InstrumentName import InstrumentName

camea_file = Path(__file__).parent.parent / 'sampledata' / 'camea2024n001486.hdf'
bambus_file = Path.home() / 'projects' / 'CoRhO4' / 'corh2o4_test.dat'
flatcone_file = Path.home() / 'projects' / 'marmot+flatcone' / '043555_pastis.nxs'
marmot_file = Path.home() / 'projects' / 'marmot+flatcone' / '054724.nxs'

@pytest.fixture(scope="module")
def multiflexx_file(tmp_path_factory):
    # No MultiFLEXX NICOS file is available: build one from the BAMBUS header, with the
    # MultiFLEXX channels ch0, ..., ch154
    lines = bambus_file.read_text().splitlines()
    header = lines[:next(i for i, line in enumerate(lines) if '### Scan data' in line) + 1]
    columns = ['sth', 'stt', 'chsum'] + [f'ch{i}' for i in range(155)] + ['events']
    rows = [' '.join([f'{sth:.3f}', '60.500', '0'] + ['1'] * 155 + ['0'])
            for sth in np.arange(-20, 21, 2.0)]
    path = tmp_path_factory.mktemp('multiflexx') / 'multiflexx_synthetic'
    path.write_text('\n'.join(header + ['#  ' + ' '.join(columns),
                                        '#  ' + ' '.join(['deg', 'deg'] + ['cts'] * 157)]
                              + rows + ['### End of NICOS data file']) + '\n')
    return path

@pytest.fixture(scope="module")
def readers():
    factory = DataReaderFactory()
    return {name: factory.get_reader_type(name) for name in InstrumentName}

def accepted_by(readers, path):
    return [name for name, reader in readers.items() if reader.can_read(path)]

def test_each_file_has_one_reader(readers, multiflexx_file):
    expected = {camea_file: InstrumentName.CAMEA,
                bambus_file: InstrumentName.BAMBUS,
                multiflexx_file: InstrumentName.MULTIFLEXX,
                flatcone_file: InstrumentName.FLATCONE,
                marmot_file: InstrumentName.MARMOT}
    for path, name in expected.items():
        assert accepted_by(readers, path) == [name], path.name

def test_unreadable_files(readers, tmp_path):
    empty = tmp_path / 'empty.dat'
    empty.write_text('')
    text = tmp_path / 'notes.txt'
    text.write_text('### Scan data\n# 1A 1B\n')
    hdf = tmp_path / 'empty.hdf'
    with h5py.File(hdf, 'w'):
        pass
    for path in [empty, text, hdf, tmp_path / 'missing.hdf']:
        assert accepted_by(readers, path) == [], path.name

def test_thales_without_marmot(readers, tmp_path):
    path = tmp_path / 'thales.nxs'
    shutil.copy(marmot_file, path)
    with h5py.File(path, 'r+') as f:
        del f['entry0/instrument_mode']
        f['entry0/instrument_mode'] = np.array([b'normal'])
    assert accepted_by(readers, path) == []

@pytest.mark.parametrize('name, path', [
    (InstrumentName.CAMEA, bambus_file),
    (InstrumentName.BAMBUS, camea_file),
    (InstrumentName.MULTIFLEXX, bambus_file),
    (InstrumentName.FLATCONE, marmot_file),
    (InstrumentName.MARMOT, flatcone_file),
])
def test_wrong_file_raises(readers, name, path):
    with pytest.raises(ValueError, match='is not a .* data file'):
        readers[name](path)

def without_data(source, path, counts):
    """A copy of a Nexus file whose counts are replaced"""
    shutil.copy(source, path)
    reader = DataReaderFactory().get_reader_type(
        InstrumentName.CAMEA if source == camea_file else InstrumentName.FLATCONE)
    with h5py.File(path, 'a') as f:
        key = next(p for p in reader.layout.field('counts').paths if p in f)
        del f[key]
        f[key] = counts
    return path

@pytest.mark.parametrize('counts', [np.zeros((0, 104, 1024), dtype='uint32'), np.array(0)],
                         ids=['no steps', 'scalar'])
def test_camea_without_data(tmp_path, counts):
    path = without_data(camea_file, tmp_path / 'empty.hdf', counts)
    with pytest.raises(ValueError, match=r'"empty.hdf" has no data \(0 scan steps\)'):
        DataReaderFactory().create(InstrumentName.CAMEA, path)

@pytest.mark.skipif(not flatcone_file.exists(), reason=f'{flatcone_file} not available')
def test_flatcone_without_data(tmp_path):
    path = without_data(flatcone_file, tmp_path / 'empty.nxs', np.zeros((31, 0)))
    with pytest.raises(ValueError, match='has no data'):
        DataReaderFactory().create(InstrumentName.FLATCONE, path)

@pytest.mark.skipif(not bambus_file.exists(), reason=f'{bambus_file} not available')
def test_nicos_without_data(tmp_path):
    # The header and column titles of a scan, but no rows
    lines = bambus_file.read_text().splitlines()
    end = next(i for i, line in enumerate(lines) if '### Scan data' in line) + 3
    path = tmp_path / 'empty.dat'
    path.write_text('\n'.join(lines[:end] + ['### End of NICOS data file']) + '\n')
    with pytest.raises(ValueError, match='has no data'):
        DataReaderFactory().create(InstrumentName.BAMBUS, path)
