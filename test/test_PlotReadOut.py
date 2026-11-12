import pytest

import warnings
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

from MJOLNIR.Data import Plotting
from MJOLNIR.Data.CutResult import COLUMNS
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')
HKL1, HKL2 = [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
ENERGY_BINS = np.linspace(-0.5, 1.5, 6)

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

@pytest.fixture(scope='module')
def dataset():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return DataSet(InstrumentName.CAMEA, files=files[:2])

def click(ax, x, y):
    """A left click at data position (x, y), through the canvas"""
    ax.figure.canvas.draw()
    pixel_x, pixel_y = ax.transData.transform((x, y))
    event = MouseEvent('button_press_event', ax.figure.canvas, pixel_x, pixel_y, button=1)
    ax.figure.canvas.callbacks.process('button_press_event', event)

def filled(data):
    """Index of a bin with points"""
    return int(np.flatnonzero(np.asarray(data[COLUMNS['binCount']]) > 0)[0])

# Plot, then the data row and the position (x, y) of one bin with points on it
def plot_1d(ds, **o):
    ax = ds.plot_cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8, **o)
    row = ax.Data.iloc[filled(ax.Data)]
    return ax, row, (row[COLUMNS['plotPosition']], row[COLUMNS['int']])

def plot_constant_q(ds, **o):
    ax = ds.plot_cut_constant_q(HKL1, **o)
    row = ax.Data.iloc[filled(ax.Data)]
    return ax, row, (row[COLUMNS['e']], row[COLUMNS['int']])

def plot_qe(ds, **o):
    ax = ds.plot_cut_qe(HKL1, HKL2, energy_bins=ENERGY_BINS, **o)
    row = ax.Data.iloc[filled(ax.Data)]
    return ax, row, (row[COLUMNS['plotPosition']], row[COLUMNS['e']])

def plot_qe_line(ds, **o):
    ax = ds.plot_cut_qe_line([HKL1, HKL2, [0.5, 0.5, 0.0]], ENERGY_BINS, **o)
    data = ax.Data[1] # Second segment: positions offset along the path
    row = data.iloc[filled(data)]
    return ax, row, (row[COLUMNS['plotPosition']] + ax.OffSets[1], row[COLUMNS['e']])

def plot_powder(ds, **o):
    ax = ds.plot_cut_powder(np.linspace(0.5, 2.0, 16), ENERGY_BINS, **o)
    row = ax.Data.iloc[filled(ax.Data)]
    return ax, row, (row[COLUMNS['qLength']], row[COLUMNS['e']])

def plot_q_plane(ds, **o):
    ax = ds.plot_q_plane(energy_min=0.0, energy_max=0.5, **o)
    row = ax.Data.iloc[filled(ax.Data)]
    return ax, row, (row[COLUMNS['qx']], row[COLUMNS['qy']])

PLOTS = dict(plot_1d=plot_1d, plot_constant_q=plot_constant_q, plot_qe=plot_qe,
             plot_qe_line=plot_qe_line, plot_powder=plot_powder, plot_q_plane=plot_q_plane)

def test_bin_text():
    row = {COLUMNS['int']: 1.25e-3, COLUMNS['intensity']: 39.0, COLUMNS['norm']: 1.5,
           COLUMNS['mon']: 1e6, COLUMNS['binCount']: 12}
    assert Plotting.bin_text(row) == ('Int = 1.2500e-03, Cts = 39, Norm = 1.500, Mon = 1000000, '
                                      'NormCount = 12')

@needs_data
@pytest.mark.parametrize('plot', PLOTS)
def test_read_out(dataset, plot):
    clicks = []
    ax, row, (x, y) = PLOTS[plot](dataset, on_click=clicks.append)
    text = ax.read_out(x, y)
    assert text == f'{ax.format_coord(x, y)}, {Plotting.bin_text(row)}'
    assert ax.bin_at(x, y).equals(row)
    click(ax, x, y) # Through pixels: the position may differ in the last digits
    assert len(clicks) == 1 and clicks[0].endswith(Plotting.bin_text(row))
    ax.on_click = None # Turned off
    click(ax, x, y)
    assert len(clicks) == 1

@needs_data
@pytest.mark.parametrize('plot', ['plot_1d', 'plot_qe', 'plot_powder', 'plot_q_plane'])
def test_read_out_outside_the_bins(dataset, plot):
    ax, _, _ = PLOTS[plot](dataset, on_click=None)
    assert ax.bin_at(1e3, 1e3) is None
    assert ax.read_out(1e3, 1e3) == ax.format_coord(1e3, 1e3)

@needs_data
def test_read_out_raw_1d_and_default(dataset, capsys):
    ax = dataset.plot_raw_1d()
    x, y = ax.results[0].data.iloc[0][['A3', COLUMNS['intensity']]]
    assert ax.read_out(x, y) == ax.format_coord(x, y) # The position, as the old plotRaw1D
    click(ax, x, y)
    assert capsys.readouterr().out.strip().startswith('A3 [degree] = ') # Printed by default

@needs_data
def test_read_out_replaced_on_shared_axes(dataset):
    first, second = [], []
    ax, _, _ = plot_1d(dataset, on_click=first.append)
    ax, _, (x, y) = plot_1d(dataset, ax=ax, on_click=second.append)
    click(ax, x, y)
    assert first == [] and len(second) == 1

@needs_data
def test_slice_viewer_read_out(dataset):
    clicks = []
    viewer = dataset.view_3d(0.05, 0.05, 0.2, on_click=clicks.append)
    values = viewer.result.grid(COLUMNS['binCount'])[:, :, viewer.index]
    i, j = np.argwhere(values > 0)[0]
    x, y = viewer.centres[0][i], viewer.centres[1][j]
    shape = values.shape + (len(viewer.centres[2]),)
    row = viewer.result.data.iloc[np.ravel_multi_index((i, j, viewer.index), shape)]
    text = viewer.read_out(x, y)
    assert text.endswith(Plotting.bin_text(row))
    click(viewer.axes, x, y)
    assert len(clicks) == 1 and clicks[0].endswith(Plotting.bin_text(row))
    viewer.cutting = True # A click while cutting starts a cut instead
    click(viewer.axes, x, y)
    assert len(clicks) == 1
    assert viewer.read_out(1e3, 1e3) == viewer._format_coord(1e3, 1e3)

@needs_data
def test_axes_match_old(dataset):
    from MJOLNIR.Data.DataSet_old import DataSet as OldDataSet
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old = OldDataSet(dataFiles=[str(f) for f in files[:2]])
        old.convertDataFile(binning=8)
    for rlu in (True, False):
        new, previous = dataset.create_q_axis(rlu=rlu), old.createQAxis(rlu=rlu, withoutOnClick=True)
        assert (new.get_xlabel(), new.get_ylabel()) == (previous.get_xlabel(), previous.get_ylabel())
        assert new.format_coord(0.3, 0.2) == previous.format_coord(0.3, 0.2)
        for axis in (0, 1):
            new, previous = dataset.create_qe_axes(axis=axis, rlu=rlu), old.createQEAxes(axis=axis, rlu=rlu)
            assert new.get_xlabel() == previous.get_xlabel()
            assert new.format_coord(0.3, 1.0) == previous.format_coord(0.3, 1.0)

@needs_data
def test_q_plane_on_created_axes(dataset):
    ax = dataset.create_q_axis()
    assert dataset.plot_q_plane(energy_min=0.0, energy_max=0.5, ax=ax, on_click=None) is ax
    with pytest.raises(ValueError, match='no data files'):
        DataSet(InstrumentName.CAMEA).create_q_axis()
