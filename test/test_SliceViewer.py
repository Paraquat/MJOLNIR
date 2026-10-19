import pytest

import numpy as np
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName
from MJOLNIR.Data.SliceViewer import SliceViewer

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
pytestmark = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')

@pytest.fixture(scope="module")
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

@pytest.fixture(scope="module")
def binned(dataset):
    return dataset.bin_3d(0.05, 0.05, 0.1)

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

def test_slices(dataset, binned):
    viewer = SliceViewer(binned, sample=dataset.sample)
    values = np.ma.masked_invalid(binned.grid('Int'))
    assert viewer.values.shape == values.shape
    assert viewer.axis == 2 and viewer.index == values.shape[2]//2 # The middle slice
    assert np.ma.allequal(viewer.mesh.get_array(), values[:, :, viewer.index].T)
    viewer.set_slice(3)
    assert np.ma.allequal(viewer.mesh.get_array(), values[:, :, 3].T)
    assert viewer.slider.val == 3
    assert viewer.slice_label() == f'E = {viewer.centres[2][3]:.3f} meV'
    viewer.set_slice(10_000) # Clamped to the last slice
    assert viewer.index == values.shape[2] - 1
    viewer.set_axis(0)
    assert viewer.qe_ax.get_visible() and not viewer.plane_ax.get_visible()
    assert np.ma.allequal(viewer.mesh.get_array(), values[viewer.index, :, :].T)
    viewer.set_axis(1)
    assert np.ma.allequal(viewer.mesh.get_array(), values[:, viewer.index, :].T)
    with pytest.raises(AttributeError, match='not recognized'):
        viewer.set_axis(3)

def test_keys(dataset, binned):
    viewer = SliceViewer(binned, sample=dataset.sample)
    start = viewer.index
    viewer._on_key(SimpleNamespace(key='up'))
    viewer._on_key(SimpleNamespace(key='up'))
    viewer._on_key(SimpleNamespace(key='left'))
    assert viewer.index == start + 1
    viewer._on_key(SimpleNamespace(key='1'))
    assert viewer.axis == 1
    viewer._on_key(SimpleNamespace(key='c'))
    assert viewer.cutting

def test_values_and_colour_scale(dataset, binned):
    counts = SliceViewer(binned, sample=dataset.sample, counts=True, log=True)
    assert np.ma.allclose(counts.values, np.ma.masked_invalid(
        np.log10(binned.grid('intensity') + 1e-20)))
    sensitivity = SliceViewer(binned, sample=dataset.sample, counts='sensitivity', vmin=0,
                              vmax=2.0)
    assert sensitivity.get_clim() == (0, 2.0) and sensitivity.mesh.get_clim() == (0, 2.0)
    sensitivity.clim_sliders[1].set_val(1.0) # The colour scale sliders
    assert sensitivity.mesh.get_clim() == (0, 1.0)
    with pytest.raises(AttributeError, match='not understood'):
        SliceViewer(binned, sample=dataset.sample, counts='unknown')
    fixed = SliceViewer(binned, sample=dataset.sample, adjustable=False)
    assert fixed.clim_sliders == []

def test_positions_in_hkl(dataset, binned):
    viewer = SliceViewer(binned, sample=dataset.sample)
    # A data point's Qx, Qy in the binning frame gives the H, K, L the reader has for it
    qx, qy = (np.asarray(a) for a in dataset.gather_q(align_to_lattice=True))
    hkl = np.array([np.asarray(dataset.gather(t)) for t in (ArrayType.H, ArrayType.K, ArrayType.L)])
    for i in np.linspace(0, len(qx) - 1, 20).astype(int):
        assert np.allclose(viewer._hkl(qx[i], qy[i]), hkl[:, i])
    viewer.set_axis(1)
    viewer.figure.canvas.draw()
    assert len(viewer.qe_ax.get_xticklabels()[0].get_text().split('\n')) == 3 # H, K, L
    centre = [c[len(c)//2] for c in viewer.centres]
    assert viewer.value_at(*centre) == pytest.approx(
        float(viewer.values[tuple(len(c)//2 for c in viewer.centres)]), nan_ok=True)
    assert np.isnan(viewer.value_at(1e3, 0.0, 0.0))

def test_without_rlu(dataset):
    binned = dataset.bin_3d(0.05, 0.05, 0.1, rlu=False)
    viewer = SliceViewer(binned, rlu=False)
    assert viewer.plane_ax.get_xlabel().startswith('$Q_x$')
    viewer.set_axis(0)
    viewer.figure.canvas.draw()
    assert len(viewer.qe_ax.get_xticklabels()[0].get_text().split('\n')) == 1
    with pytest.raises(AttributeError, match='sample is needed'):
        SliceViewer(binned)
    with pytest.raises(ValueError, match='grid'):
        SliceViewer(dataset.cut_1d([-1, 0, 0], [0, 1, 0], 0.1, 0.02, -1.0, 0.8),
                    sample=dataset.sample)

def test_cuts_go_to_on_cut(dataset, binned):
    received = []
    viewer = SliceViewer(binned, sample=dataset.sample,
                         on_cut=lambda kind, parameters: received.append((kind, parameters)) or kind)
    x, y = viewer.centres[0], viewer.centres[1]
    assert viewer.cut((x[3], y[3]), (x[-3], y[-3])) == '1d'
    kind, parameters = received[-1]
    assert np.allclose(parameters['q1'], viewer._hkl(x[3], y[3])) and parameters['rlu']
    assert (parameters['energy_min'], parameters['energy_max']) == \
        tuple(viewer.edges[2][viewer.index:viewer.index + 2])
    assert len(dataset.cut_1d(**parameters)) > 0 # The parameters are cut_1d's
    viewer.set_axis(1)
    energies = viewer.edges[2]
    viewer.cut((x[10], energies[2]), (x[10], energies[-2])) # A click: constant Q
    kind, parameters = received[-1]
    assert kind == 'constant_q'
    assert np.allclose(parameters['q'], viewer._hkl(x[10], viewer.centres[1][viewer.index]))
    assert len(dataset.cut_constant_q(**parameters)) > 0
    viewer.cut((x[5], energies[2]), (x[-5], energies[10])) # A line: Q-E cut
    kind, parameters = received[-1]
    assert kind == 'qe' and np.allclose(parameters['energy_bins'], energies[2:11])
    assert len(dataset.cut_qe(**parameters)) > 0
    assert viewer.cuts == ['1d', 'constant_q', 'qe']

def test_default_cuts(dataset, binned):
    viewer = dataset.view_3d(0.05, 0.05, 0.1)
    x, y = viewer.centres[0], viewer.centres[1]
    ax = viewer.cut((x[3], y[3]), (x[-3], y[-3]))
    assert ax.result is not None and ax is not viewer.axes # Plotted by the data set
    viewer.set_axis(0)
    energies = viewer.edges[2]
    ax = viewer.cut((y[10], energies[2]), (y[10], energies[-2]))
    assert len(ax.result) > 0 and ax.get_xlabel() == 'E [meV]'
    alone = SliceViewer(binned, sample=dataset.sample)
    with pytest.warns(UserWarning, match='neither on_cut nor a data set'):
        assert alone.cut((x[3], y[3]), (x[-3], y[-3])) is None

def test_spurions(dataset):
    peaks = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
    viewer = dataset.view_3d(0.05, 0.05, 0.1, spurion_peaks=peaks)
    assert set(viewer.spurions) == {'Monochromator', 'Analyser'}
    # Elastic spurions of both origins sit on the Bragg peaks
    Ei = dataset._readers[0].Ei[0]
    for origin in ('Monochromator', 'Analyser'):
        positions = viewer.frame_sample.CurratAxe(Ei=[Ei], Ef=[Ei], Bragg=peaks,
                                                  spurionType=origin, HKL=False)[:, 0, 0, :2]
        assert np.allclose([viewer._hkl(*p) for p in positions], peaks)
    # Markers on a slice: the spurions within the slice
    points = viewer.spurions['Monochromator']
    index = int(np.argmin(np.abs(viewer.centres[2] - np.median(points[:, 2]))))
    viewer.set_slice(index)
    width = np.diff(viewer.edges[2])[index]
    expected = np.abs(points[:, 2] - viewer.centres[2][index]) <= width/2
    assert len(viewer._spurion_lines[0].get_xdata()) == expected.sum()

def test_saving(dataset, binned, tmp_path):
    viewer = SliceViewer(binned, sample=dataset.sample)
    viewer.to_csv(tmp_path / 'slice.csv')
    lines = (tmp_path / 'slice.csv').read_text().splitlines()
    assert lines[0] == 'qx,qy,energy,value'
    assert len(lines) == 1 + viewer.values.shape[0]*viewer.values.shape[1]
    viewer.set_axis(0)
    current = viewer.index
    saved = viewer.save_slices(tmp_path / 'slices')
    assert len(saved) == viewer.values.shape[0] and all(path.exists() for path in saved)
    assert saved[0].name == 'qx_000.png' and viewer.index == current

def test_drawing_a_cut_with_the_mouse(dataset, binned):
    from matplotlib.backend_bases import MouseEvent, KeyEvent
    received = []
    viewer = SliceViewer(binned, sample=dataset.sample,
                         on_cut=lambda kind, parameters: received.append((kind, parameters)))
    canvas = viewer.figure.canvas
    canvas.draw()
    x, y = viewer.centres[0], viewer.centres[1]
    start, end = (x[5], y[5]), (x[-5], y[-5])
    pixels = [viewer.axes.transData.transform(point) for point in (start, end)]
    canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', canvas, 'c'))
    for name, (px, py) in [('button_press_event', pixels[0]),
                           ('motion_notify_event', pixels[1]),
                           ('button_release_event', pixels[1])]:
        canvas.callbacks.process(name, MouseEvent(name, canvas, px, py, button=1))
    kind, parameters = received[-1]
    assert kind == '1d'
    assert np.allclose(parameters['q1'], viewer._hkl(*start), atol=1e-6)
    assert np.allclose(parameters['q2'], viewer._hkl(*end), atol=1e-6)
    assert viewer._cut_line is None # The rubber band line is removed
