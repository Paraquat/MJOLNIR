import pytest

import h5py
import shutil
import numpy as np

from MJOLNIR.Data.DataReaderFactory import DataReaderFactory
from MJOLNIR.Data.InstrumentName import InstrumentName


@pytest.fixture(scope="module")
def multiflexx_file(bambus_file, tmp_path_factory):
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

def test_each_file_has_one_reader(camea_2024_files, bambus_file, flatcone_file, marmot_file, readers, multiflexx_file):
    expected = {camea_2024_files[0]: InstrumentName.CAMEA,
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

def test_thales_without_marmot(marmot_file, readers, tmp_path):
    path = tmp_path / 'thales.nxs'
    shutil.copy(marmot_file, path)
    with h5py.File(path, 'r+') as f:
        del f['entry0/instrument_mode']
        f['entry0/instrument_mode'] = np.array([b'normal'])
    assert accepted_by(readers, path) == []

@pytest.mark.parametrize('name, data', [ # Data: the fixture with a file of another instrument
    (InstrumentName.CAMEA, 'bambus_file'),
    (InstrumentName.BAMBUS, 'camea_2024_files'),
    (InstrumentName.MULTIFLEXX, 'bambus_file'),
    (InstrumentName.FLATCONE, 'marmot_file'),
    (InstrumentName.MARMOT, 'flatcone_file'),
])
def test_wrong_file_raises(request, readers, name, data):
    path = request.getfixturevalue(data)
    if isinstance(path, list):
        path = path[0]
    with pytest.raises(ValueError, match='is not a .* data file'):
        readers[name](path)

def without_data(source, path, counts):
    """A copy of a Nexus file whose counts are replaced"""
    shutil.copy(source, path)
    reader = DataReaderFactory().get_reader_type(
        InstrumentName.CAMEA if source.suffix == '.hdf' else InstrumentName.FLATCONE)
    with h5py.File(path, 'a') as f:
        key = next(p for p in reader.layout.field('counts').paths if p in f)
        del f[key]
        f[key] = counts
    return path

@pytest.mark.parametrize('counts', [np.zeros((0, 104, 1024), dtype='uint32'), np.array(0)],
                         ids=['no steps', 'scalar'])
def test_camea_without_data(camea_2024_files, tmp_path, counts):
    path = without_data(camea_2024_files[0], tmp_path / 'empty.hdf', counts)
    with pytest.raises(ValueError, match=r'"empty.hdf" has no data \(0 scan steps\)'):
        DataReaderFactory().create(InstrumentName.CAMEA, path)

def test_flatcone_without_data(flatcone_file, tmp_path):
    path = without_data(flatcone_file, tmp_path / 'empty.nxs', np.zeros((31, 0)))
    with pytest.raises(ValueError, match='has no data'):
        DataReaderFactory().create(InstrumentName.FLATCONE, path)

def test_nicos_without_data(bambus_file, tmp_path):
    # The header and column titles of a scan, but no rows
    lines = bambus_file.read_text().splitlines()
    end = next(i for i, line in enumerate(lines) if '### Scan data' in line) + 3
    path = tmp_path / 'empty.dat'
    path.write_text('\n'.join(lines[:end] + ['### End of NICOS data file']) + '\n')
    with pytest.raises(ValueError, match='has no data'):
        DataReaderFactory().create(InstrumentName.BAMBUS, path)
