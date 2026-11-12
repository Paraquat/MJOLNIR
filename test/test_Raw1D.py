import pytest

import h5py
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.DataReader import SampleEnvironment
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')
OTHER_DATA = Path.home() / 'projects'

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

@needs_data
def test_reader_selection_and_scan():
    reader = CameaDataReader(files[0])
    assert (reader.detector_selection, reader.analyser_selection) == (58, 4)
    assert list(reader.scan_parameters) == ['A3'] and list(reader.scan_units) == ['degree']
    assert np.array_equal(reader.scan_values[0], reader.A3)

@needs_data
def test_reader_time_and_scan_command():
    reader = CameaDataReader(files[0])
    with h5py.File(files[0]) as f:
        assert np.array_equal(reader.time, f['entry/control/time'][()]) # Seconds per step
        assert reader._metadata['scanCommand'] == f['entry/scancommand'][0].decode()
    assert reader.time.shape == (reader.n_steps,)
    assert 'CAMEAscan' in reader._metadata['scanCommand']
    values = CameaDataReader.shallow_read(files[0], ['time', 'scanCommand'])
    assert np.array_equal(values['time'], reader.time)
    assert values['scanCommand'] == reader._metadata['scanCommand']

@needs_data
def test_raw_counts():
    reader = CameaDataReader(files[0])
    reader.set_instrument_calibration(8)
    calibration = reader.calibration
    low, high = reader.get_instrument_calibration(1).bound.reshape(104, 8, 2)[58, 4]
    reader.binning = 8 # get_instrument_calibration set it; as before the call above
    expected = reader.counts[:, 58, low:high].sum(axis=1)
    assert np.array_equal(reader.raw_counts(), expected) # The file's selection
    assert reader.binning == 8 and reader.calibration is calibration # State kept
    low, high = reader.get_instrument_calibration(1).bound.reshape(104, 8, 2)[0, 7]
    assert np.array_equal(reader.raw_counts(0, 7), reader.counts[:, 0, low:high].sum(axis=1))
    for detector, analyser in [(104, 0), (0, 8), (-1, 0)]:
        with pytest.raises(ValueError, match='out of range'):
            reader.raw_counts(detector, analyser)

@needs_data
def test_cut_raw_1d():
    dataset = DataSet(InstrumentName.CAMEA, files=files[:2])
    results = dataset.cut_raw_1d()
    assert [r.parameters['file'] for r in results] == [f.name for f in files[:2]]
    first = results[0]
    assert list(first.data.columns) == ['A3', 'intensity', 'Monitor']
    assert (first.parameters['detector'], first.parameters['analyser']) == (58, 4)
    assert np.array_equal(first.data['intensity'], dataset._readers[0].raw_counts())
    per_file = dataset.cut_raw_1d(detector=[10, 20], analyser=3)
    assert [r.parameters['detector'] for r in per_file] == [10, 20]
    with pytest.raises(ValueError, match='2 data files'):
        dataset.cut_raw_1d(detector=[1, 2, 3])
    with pytest.raises(ValueError, match='derived data'):
        (dataset - dataset).cut_raw_1d()

@needs_data
def test_plot_raw_1d(tmp_path):
    dataset = DataSet(InstrumentName.CAMEA, files=files[:2])
    ax = dataset.plot_raw_1d(detector=50, analyser=2)
    assert len(ax.collections) == 2
    for collection, result in zip(ax.collections, ax.results):
        assert np.allclose(collection.get_offsets(),
                           result.data[['A3', 'intensity']].to_numpy())
    assert ax.get_xlabel() == 'A3 [degree]' and ax.get_ylabel() == 'Int [counts]'
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [f.name for f in files[:2]]
    ax.to_csv(tmp_path / 'raw.csv')
    assert len((tmp_path / 'raw.csv').read_text().splitlines()) == 1 + sum(map(len, ax.results))
    named = dataset.plot_raw_1d(legend=['first', 'second'])
    assert [t.get_text() for t in named.get_legend().get_texts()] == ['first', 'second']
    with pytest.raises(AttributeError, match='legends'):
        dataset.plot_raw_1d(legend=['only one'])
    assert dataset.plot_raw_1d(legend=False).get_legend() is None

def fake_camea(tmp_path, scanvars: bytes, **arrays):
    """A reader object on a minimal CAMEA-like file, to test the scan variable logic"""
    path = tmp_path / 'fake.hdf'
    with h5py.File(path, 'w') as f:
        f['entry/scanvars'] = np.array([scanvars])
        f['entry/data/data'] = np.zeros((3, 2, 2)) # Excluded: the counts
    reader = CameaDataReader.__new__(CameaDataReader)
    reader._file = h5py.File(path)
    reader.file = path
    reader.environment = SampleEnvironment(temperature=arrays.get('temperature'))
    reader.Ei = arrays.get('Ei', np.full(3, 5.0))
    reader.A3 = arrays.get('A3', np.full(3, 10.0))
    reader.twotheta = np.full(3, -40.0)
    return reader

def test_scan_variable_fallbacks(tmp_path):
    # Manual scan (no scanvars): the first quantity that changes
    reader = fake_camea(tmp_path, b'', Ei=np.array([5.0, 5.1, 5.2]))
    parameters, values, units = reader.get_scan_parameters()
    assert list(parameters) == ['Ei'] and list(units) == ['meV']
    assert np.allclose(values[0], [5.0, 5.1, 5.2])
    reader._file.close()
    reader = fake_camea(tmp_path, b'', temperature=np.array([2.0, 3.0, 4.0]),
                        Ei=np.array([5.0, 5.1, 5.2]))
    assert list(reader.get_scan_parameters()[0]) == ['T'] # Temperature first, as the old code
    reader._file.close()
    # Unknown and Q-scan variables are left out with a warning
    reader = fake_camea(tmp_path, b'magnet,CAMEA,')
    with pytest.warns(UserWarning) as warnings:
        assert len(reader.get_scan_parameters()[0]) == 0
    assert any('magnet' in str(w.message) for w in warnings)
    assert any('qcscan' in str(w.message) for w in warnings)
    reader._file.close()

@pytest.mark.skipif(not (OTHER_DATA / 'CoRhO4' / 'corh2o4_test.dat').exists(),
                    reason='CoRhO4 Bambus file not available')
def test_raw_counts_one_channel_per_analyser():
    from MJOLNIR.Data.BambusDataReader import BAMBUSDataReader
    reader = BAMBUSDataReader(OTHER_DATA / 'CoRhO4' / 'corh2o4_test.dat')
    assert (reader.detector_selection, reader.analyser_selection) == (0, 0)
    assert np.array_equal(reader.raw_counts(10, 2), reader.counts[:, 10, 2])

@pytest.mark.skipif(not (OTHER_DATA / 'marmot+flatcone' / '043555_pastis.nxs').exists(),
                    reason='Flatcone file not available')
def test_raw_counts_flatcone():
    """The old cutRaw1D failed on Flatcone files (IndexError)"""
    dataset = DataSet(InstrumentName.FLATCONE,
                      files=[OTHER_DATA / 'marmot+flatcone' / '043555_pastis.nxs'])
    result = dataset.cut_raw_1d(detector=30)[0]
    assert np.array_equal(result.data['intensity'], dataset._readers[0].counts[:, 30, 0])
    assert result.parameters['scan_parameters'] == ['a3p']

@needs_data
def test_reader_repr():
    reader = CameaDataReader(files[0])
    assert repr(reader) == f"CameaDataReader('{files[0].name}', 135 steps, Ei 4.00 meV)"
    reader.set_instrument_calibration(8)
    assert repr(reader).endswith('Ei 4.00 meV, binning 8)')
    reader.Ei = np.linspace(4.0, 5.0, reader.n_steps) # An Ei scan: the range
    assert 'Ei 4.00-5.00 meV' in repr(reader)
    assert repr(CameaDataReader.__new__(CameaDataReader)) == "CameaDataReader('?')" # Half built
