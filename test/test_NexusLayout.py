import pytest

import h5py
from types import SimpleNamespace
import numpy as np
from pathlib import Path

from MJOLNIR.Data.NexusLayout import Field, Group, NexusLayout, CameaLayout
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.NexusDataReader import NexusDataReader

SAMPLE_DATA = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))

def test_field_format_keeps_unknown_placeholders():
    field = Field('{entry}/{instrument}/calib{binning}/width', aliases=('{entry}/w',))
    filled = field.format(entry='entry0', instrument='IN20')
    assert filled.path == 'entry0/IN20/calib{binning}/width'
    assert filled.aliases == ('entry0/w',)
    assert filled.paths == ('entry0/IN20/calib{binning}/width', 'entry0/w')
    assert field.path == '{entry}/{instrument}/calib{binning}/width' # Unchanged

def test_camea_paths():
    assert CameaLayout.path('counts') == 'entry/CAMEA/detector/data'
    assert CameaLayout.field('counts').paths == ('entry/CAMEA/detector/data',
                                                 'entry/CAMEA/detector/counts')
    assert CameaLayout.path('title') == 'entry/title' # Metadata
    assert CameaLayout.field('temperature').paths[-1] == 'entry/sample/T' # Environment
    assert CameaLayout.group('instrument') == Group('entry/CAMEA', 'NXinstrument')
    assert CameaLayout.calibration_group_for(3) == Group('entry/CAMEA/calib3', 'NXcollection')
    assert CameaLayout.calibration_fields(8)['Ef'].path == 'entry/CAMEA/calib8/final_energy'
    with pytest.raises(KeyError):
        CameaLayout.field('notAField')

def test_calibration_keys_match_instrument_calibration():
    from dataclasses import fields
    from MJOLNIR.Data.DataReader import InstrumentCalibration
    assert set(CameaLayout.calibration) == {f.name for f in fields(InstrumentCalibration)}

def test_no_placeholders_left():
    for field in CameaLayout.all_fields().values():
        assert all('{' not in path for path in field.paths)
    for group in CameaLayout.all_groups().values():
        assert '{' not in group.path
    for field in CameaLayout.calibration_fields(1).values():
        assert '{' not in field.path

def test_subclass_changes_entry():
    """Templates are kept, so a subclass can move the whole layout"""
    class Moved(CameaLayout):
        entry = 'entry1'
    assert Moved.path('counts') == 'entry1/CAMEA/detector/data'
    assert Moved.calibration_fields(1)['A4'].path == 'entry1/CAMEA/calib1/a4offset'
    assert CameaLayout.path('counts') == 'entry/CAMEA/detector/data'

def test_subclass_checks():
    with pytest.raises(TypeError, match='must define instrument'):
        class NoInstrument(NexusLayout):
            pass
    with pytest.raises(TypeError, match='defined twice'):
        class Repeated(NexusLayout):
            instrument = 'X'
            metadata = {**NexusLayout.metadata, 'A3': Field('{entry}/a3')}
    with pytest.raises(TypeError, match='unknown placeholder'):
        class Misspelt(NexusLayout):
            instrument = 'X'
            fields = {'A3': Field('{entri}/a3')}
    with pytest.raises(TypeError, match='unknown placeholder'):
        class BinningOutsideCalibration(NexusLayout):
            instrument = 'X'
            fields = {'A3': Field('{entry}/calib{binning}/a3')}

def test_reader_tries_aliases(tmp_path):
    """getValue takes the first of a field's paths in the file: old CAMEA files store the
    counts in detector/counts"""
    with h5py.File(tmp_path / 'old.hdf', 'w') as f:
        f['entry/CAMEA/detector/counts'] = np.ones((2, 3, 4))
        f['entry/sample/T'] = [1.5, 2.5]
        f['entry/sample/temperature'] = [3.5, 4.5]
    reader = SimpleNamespace(layout=CameaLayout, _file=h5py.File(tmp_path / 'old.hdf'))
    assert NexusDataReader.getValue(reader, 'counts').shape == (2, 3, 4)
    assert NexusDataReader.getValue(reader, 'temperature')[0] == 3.5 # Main path first
    assert NexusDataReader.getValue(reader, 'sample').name == '/entry/sample' # A group
    assert NexusDataReader.getValue(reader, 'Ei') is None # In the layout, not in the file
    assert NexusDataReader.getValue(reader, 'notAField') is None
    reader._file.close()

@pytest.mark.skipif(not SAMPLE_DATA, reason='sampledata/camea2024*.hdf not available')
@pytest.mark.parametrize('data_file', SAMPLE_DATA[:2], ids=lambda p: p.name)
def test_camea_matches_files(data_file):
    """Groups, units and dtypes of every main path present in a NICOS file"""
    def units(dataset):
        value = dataset.attrs.get('units')
        return value.decode() if isinstance(value, bytes) else value

    with h5py.File(data_file) as f:
        for group in CameaLayout.all_groups().values():
            if group.path in f:
                assert f[group.path].attrs['NX_class'].decode() == group.nx_class, group.path
        fields = list(CameaLayout.all_fields().items())
        for binning in (1, 3, 5, 8):
            group = CameaLayout.calibration_group_for(binning)
            assert f[group.path].attrs['NX_class'].decode() == group.nx_class
            fields += CameaLayout.calibration_fields(binning).items()
        found = 0
        for key, field in fields:
            if field.path not in f:
                continue
            found += 1
            dataset = f[field.path]
            assert units(dataset) == field.units, key
            if field.dtype is not None:
                assert dataset.dtype == np.dtype(field.dtype), key
        assert f[CameaLayout.path('counts')].ndim == 3
    assert found > 60

@pytest.mark.skipif(not SAMPLE_DATA, reason='sampledata/camea2024*.hdf not available')
def test_camea_reader_uses_layout():
    data_file = SAMPLE_DATA[0]
    assert CameaDataReader.can_read(data_file)
    reader = CameaDataReader(data_file)
    assert reader.layout is CameaLayout
    assert reader.possible_binnings == [1, 3, 5, 8]
    calibration = reader.get_instrument_calibration(3)
    with h5py.File(data_file) as f:
        assert np.array_equal(calibration.Ef, f['entry/CAMEA/calib3/final_energy'][()])
        assert np.array_equal(calibration.bound, f['entry/CAMEA/calib3/boundaries'][()])
        assert np.array_equal(reader.environment.temperature, f['entry/sample/temperature'][()])
        assert np.array_equal(reader.monitor2, f['entry/monitor_2/data'][()])
    assert reader.environment.units['temperature'] == 'K'
    assert reader._metadata['sampleName'] == 'TbFeO3'
