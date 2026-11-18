import pytest

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.CommandLineScripts import _tools
from MJOLNIR.CommandLineScripts import MJOLNIRHistory, MJOLNIR3DView, MJOLNIRCalibrationInspector
from MJOLNIR.Data import Plotting
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.InstrumentName import InstrumentName
from MJOLNIR.Data.SliceViewer import SliceViewer


@pytest.fixture(autouse=True)
def settings_file(tmp_path, monkeypatch):
    """Keep the scripts' remembered files (--reuse) out of the user's home folder"""
    path = tmp_path / '.MJOLNIRsettings'
    monkeypatch.setattr(_tools, 'settingsFile', str(path))
    yield path
    plt.close('all')

@pytest.mark.parametrize('module', [MJOLNIRHistory, MJOLNIR3DView, MJOLNIRCalibrationInspector])
def test_instrument_option(module):
    assert module.parser.parse_args(['file']).instrument == 'CAMEA'
    assert module.parser.parse_args(['file', '-i', 'MultiFLEXX']).instrument == 'MultiFLEXX'
    assert _tools.instrumentName(module.parser.parse_args(['file', '--instrument', 'Marmot'])) == InstrumentName.MARMOT
    with pytest.raises(SystemExit):
        module.parser.parse_args(['file', '-i', 'NotAnInstrument'])

## History

def test_History_lines(camea_2024_files, capsys):
    MJOLNIRHistory.main([str(camea_2024_files[0]), str(camea_2024_files[1])])
    lines = capsys.readouterr().out
    for path in camea_2024_files[:2]:
        values = CameaDataReader.shallow_read(path, ['scanCommand', 'title'])
        assert path.name+': '+values['scanCommand']+'\t'+values['title']+'\n' in lines

def test_History_wrong_format(capsys, tmp_path):
    path = tmp_path / 'notdata.hdf'
    path.write_text('not a data file')
    MJOLNIRHistory.main([str(path)])
    assert capsys.readouterr().out == 'notdata.hdf not correct format\n'

def test_History_save_and_reuse(camea_2024_files, capsys, tmp_path):
    save = tmp_path / 'history'
    MJOLNIRHistory.main([str(camea_2024_files[0]), '-s', str(save)])
    first = capsys.readouterr().out
    assert (tmp_path / 'history.txt').read_text() == first

    MJOLNIRHistory.main(['-r', '-s', str(save)]) # The same file again; the old history is kept
    assert 'File already exists' in capsys.readouterr().out
    assert (tmp_path / 'history_old.txt').read_text() == first
    assert (tmp_path / 'history.txt').read_text() == first

## Calibration inspector

def test_CalibrationInspector_saves_all_plots(camea_2024_files, tmp_path):
    args = MJOLNIRCalibrationInspector.parser.parse_args([str(camea_2024_files[0]), '-s', str(tmp_path / 'plots'), '-b', '1'])
    axes = MJOLNIRCalibrationInspector.inspect(args)
    assert list(axes) == ['A4', 'Normalization', 'Ef', 'EfOverview']
    names = sorted(p.name for p in (tmp_path / 'plots').iterdir())
    assert names == ['Final_Energy_Individual_1.png', 'Final_Energy_Overview_1.png',
                     'Instrument_calibration_1.png', 'Normalization_1.png']

def test_CalibrationInspector_one_plot(camea_2024_files, tmp_path):
    save = tmp_path / 'ef.png'
    args = MJOLNIRCalibrationInspector.parser.parse_args([str(camea_2024_files[0]), '-p', 'ef', 'unknown', '-s', str(save)])
    with pytest.warns(UserWarning, match='"unknown" not understood'):
        axes = MJOLNIRCalibrationInspector.inspect(args)
    assert list(axes) == ['Ef']
    assert save.is_file()

def test_CalibrationInspector_wrong_binning(camea_2024_files):
    args = MJOLNIRCalibrationInspector.parser.parse_args([str(camea_2024_files[0]), '-b', '2'])
    with pytest.raises(ValueError, match='Binning 2 not in the possible binnings'):
        MJOLNIRCalibrationInspector.inspect(args)

def test_calibration_plots_show_the_calibration(camea_2024_files):
    reader = CameaDataReader(camea_2024_files[0])
    calibration = reader.get_instrument_calibration(8)
    A4 = calibration.A4.reshape(reader.n_detectors, -1)
    normalization = (calibration.amp * calibration.width * np.sqrt(2 * np.pi)).reshape(reader.n_detectors, -1)

    ax = Plotting.plot_calibration_a4(calibration, reader.n_detectors)
    assert len(ax.collections) == reader.n_detectors
    assert np.allclose(ax.collections[3].get_offsets()[:, 0], -A4[3], equal_nan=True)
    assert np.allclose(ax.collections[3].get_array(), normalization[3], equal_nan=True)

    ax = Plotting.plot_calibration_normalization(calibration, reader.n_detectors)
    assert np.allclose(ax.images[0].get_array(), normalization.T, equal_nan=True)

    ax = Plotting.plot_calibration_ef_overview(calibration, reader.n_detectors)
    assert ax.images[0].get_array().shape == (8 * reader.n_analysers, reader.n_detectors)

    ax = Plotting.plot_calibration_ef(calibration, reader.n_detectors)
    assert np.allclose(ax.collections[5].get_offsets()[:, 1], calibration.Ef.reshape(reader.n_detectors, -1)[5], equal_nan=True)

## 3D view

def test_3DView(camea_2024_files, settings_file):
    args = MJOLNIR3DView.parser.parse_args([str(camea_2024_files[0]), '-d', '0.1', '0.1', '0.2', '-m', '0', '-M', '1e-5'])
    dataSet, viewer = MJOLNIR3DView.view(args)
    assert isinstance(viewer, SliceViewer)
    assert viewer._clim == (0.0, 1e-5)
    assert {reader.binning for reader in dataSet} == {8}

    args = MJOLNIR3DView.parser.parse_args(['-r', '-b', '1', '-d', '0.1', '0.1', '0.2'])
    dataSet, viewer = MJOLNIR3DView.view(args)
    assert [reader.file for reader in dataSet] == [camea_2024_files[0]]
    assert {reader.binning for reader in dataSet} == {1}
