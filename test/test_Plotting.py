import pytest

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data import Cuts, Plotting
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
HKL1, HKL2 = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

@pytest.fixture(scope="module")
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

@pytest.fixture(scope="module")
def result(dataset):
    qx, qy = dataset.gather_q()
    gather = lambda arr_type: np.asarray(dataset.gather(arr_type))
    q1, q2 = dataset.convert_to_qxqy([HKL1, HKL2])
    return Cuts.cut_qe(qx=np.asarray(qx), qy=np.asarray(qy), energy=gather(ArrayType.ENERGY),
                       intensity=gather(ArrayType.INTENSITY),
                       normalization=gather(ArrayType.NORMALIZATION),
                       monitor=gather(ArrayType.MONITOR), q1=q1, q2=q2, width=0.1,
                       min_pixel=0.02, energy_min=-1.0, energy_max=0.8, energy_step=0.1,
                       sample=dataset.sample)

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

def test_plot_cut_qe(dataset, result):
    ax = Plotting.plot_cut_qe(result, sample=dataset.sample, colorbar=True)
    mesh = ax.pmeshs[0]
    assert mesh.get_array().shape == (18, 82) # Energy bins x Q bins
    assert np.ma.allclose(mesh.get_array(),
                          np.asarray(result.data['Int']).reshape(18, 82), masked_equal=True)
    # x runs from 0 at the start of the cut to 1 at its end; y is the lower energy edge
    length = np.linalg.norm(result.parameters['q2'] - result.parameters['q1'])
    assert ax.X[0] == pytest.approx(0.5*0.02/length) # Middle of the first Q bin
    # Bins run one pixel past q2, so that the whole cut is covered
    assert np.all(np.diff(ax.X) > 0) and 1 < ax.X[-1] < 1 + 0.02/length
    assert np.allclose(ax.Y, result.bins['energy'][:-1])
    assert np.allclose(ax.calculatePosition(0), HKL1) and np.allclose(ax.calculatePosition(1), HKL2)
    assert 'BinDistance' in result.data
    assert ax.get_ylabel() == 'E [meV]'
    assert 'Q_h' in ax.get_xlabel()
    ax.set_clim(0, 0.025)
    assert ax.get_clim() == (0, 0.025)
    assert ax.colorbar is not None

def test_plot_options(dataset, result):
    counts = Plotting.plot_cut_qe(result, sample=dataset.sample, counts=True).pmeshs[0]
    assert np.nansum(counts.get_array()) == result.data['intensity'].sum()
    sensitivity = Plotting.plot_cut_qe(result, sample=dataset.sample, counts='sensitivity')
    assert sensitivity.pmeshs[0].get_array().max() > 0
    log = Plotting.plot_cut_qe(result, sample=dataset.sample, log=True).pmeshs[0]
    linear = np.asarray(result.data['Int']).reshape(18, 82)
    assert np.ma.allclose(log.get_array(), np.log10(linear + 1e-20), masked_equal=True)
    in_q = Plotting.plot_cut_qe(result, rlu=False)
    assert 'Q_x' in in_q.get_xlabel()
    with pytest.raises(AttributeError): # H, K, L need the sample
        Plotting.plot_cut_qe(result)
    with pytest.raises(AttributeError):
        Plotting.plot_cut_qe(result, sample=dataset.sample, counts='photons')

def test_given_axes_and_csv(dataset, result, tmp_path):
    _, ax = plt.subplots()
    returned = Plotting.plot_cut_qe(result, sample=dataset.sample, ax=ax)
    assert returned is ax
    ax.to_csv(tmp_path / 'cut.csv')
    lines = (tmp_path / 'cut.csv').read_text().splitlines()
    assert lines[0].startswith('# CSV generated from MJOLNIR')
    assert len(lines) == 2 + len(result) # Comment, header, one line per bin

def test_tick_labels(dataset, result):
    ax = Plotting.plot_cut_qe(result, sample=dataset.sample)
    ax.get_figure().canvas.draw()
    first = ax.get_xticklabels()[0].get_text().split('\n')
    assert len(first) == 3 # H, K and L of the tick

# Constant-energy planes

@pytest.fixture(scope="module")
def plane(dataset):
    from MJOLNIR.Data.Coordinates import align_to_lattice
    qx, qy = dataset.gather_q(align_to_lattice=True)
    gather = lambda arr_type: np.asarray(dataset.gather(arr_type))
    return Cuts.cut_q_plane(np.asarray(qx), np.asarray(qy), gather(ArrayType.ENERGY),
                            gather(ArrayType.INTENSITY), gather(ArrayType.NORMALIZATION),
                            gather(ArrayType.MONITOR), energy_min=-0.2, energy_max=0.2,
                            x_bin_tolerance=0.05, y_bin_tolerance=0.05,
                            sample=align_to_lattice(dataset.sample))

def test_plot_q_plane(dataset, plane, tmp_path):
    ax = Plotting.plot_q_plane(plane, sample=dataset.sample, colorbar=True)
    assert len(ax.pmeshs) == 1
    values = np.asarray(ax.pmeshs[0].get_array())
    assert np.array_equal(values.flatten(), np.asarray(plane.data['Int']), equal_nan=True)
    assert ax.Qx[0].shape == (50, 44) # Bin edges
    assert ax.get_clim() == (np.nanmin(values), np.nanmax(values)) # Default colour scale
    ax.set_clim(0, 0.025)
    assert ax.get_clim() == (0, 0.025)
    assert ax.get_xlabel().endswith('[RLU]') and ax.get_ylabel().endswith('[RLU]')
    assert ax.get_xlim() == (plane.bins['qx'][0][0], plane.bins['qx'][0][-1])
    ax.to_csv(tmp_path / 'plane.csv')
    lines = (tmp_path / 'plane.csv').read_text().splitlines()
    assert lines[0].endswith('Shape of data is (49, 43)')
    assert len(lines) == 2 + len(plane)

def test_plot_q_plane_options(dataset, plane):
    log = Plotting.plot_q_plane(plane, sample=dataset.sample, log=True)
    assert np.allclose(np.asarray(log.pmeshs[0].get_array()).flatten(),
                       np.log10(1e-20 + np.asarray(plane.data['Int'])), equal_nan=True)
    in_q = Plotting.plot_q_plane(plane, rlu=False) # Plain Qx, Qy axes need no sample
    assert in_q.get_xlabel() == 'Qx [1/AA]'
    with pytest.raises(AttributeError): # Reciprocal lattice axes need the sample
        Plotting.plot_q_plane(plane)
    # 3D axes (the old plotQPlane's 3D layers) are refused, not drawn flat
    with pytest.raises(ValueError, match='not 3D axes'):
        Plotting.plot_q_plane(plane, ax=plt.figure().add_subplot(projection='3d'))

# 1D cuts

@pytest.fixture(scope="module")
def cut_1d(dataset):
    return dataset.cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8)

def test_plot_cut_1d(dataset, cut_1d):
    ax = Plotting.plot_cut_1d(cut_1d, sample=dataset.sample)
    data = ax.Data
    x, y = ax.lines[0].get_data()
    assert np.allclose(x, data['BinDistance']) and np.allclose(y, data['Int'], equal_nan=True)
    assert data['BinDistance'].iloc[0] == pytest.approx(0.0, abs=0.1) # Starts near q1 (0)
    assert data['BinDistance'].iloc[-1] == pytest.approx(1.0, abs=0.1) # Ends near q2 (1)
    assert ax.energy == pytest.approx(-0.1) # Mean energy, shown in the tick labels
    assert ax.get_ylabel() == '$I$ [arb.u.]'
    ax.get_figure().canvas.draw()
    assert len(ax.get_xticklabels()[0].get_text().split('\n')) == 4 # H, K, L and E
    assert ax.result is cut_1d

def test_plot_cut_1d_options(dataset, cut_1d, tmp_path):
    counts = Plotting.plot_cut_1d(cut_1d, sample=dataset.sample, counts=True, show_energy=False)
    assert np.allclose(counts.lines[0].get_ydata(), cut_1d.data['intensity'])
    counts.get_figure().canvas.draw()
    assert len(counts.get_xticklabels()[0].get_text().split('\n')) == 3 # No energy
    sensitivity = Plotting.plot_cut_1d(cut_1d, sample=dataset.sample, counts='sensitivity')
    assert np.allclose(sensitivity.lines[0].get_ydata(),
                       cut_1d.data['Normalization']/cut_1d.data['BinCount'], equal_nan=True)
    with pytest.raises(AttributeError, match='not understood'):
        Plotting.plot_cut_1d(cut_1d, sample=dataset.sample, counts='unknown')
    with pytest.raises(AttributeError, match='sample is needed'):
        Plotting.plot_cut_1d(cut_1d)
    qxqy = Plotting.plot_cut_1d(cut_1d, rlu=False)
    assert 'Q_x' in qxqy.get_xlabel()
    qxqy.to_csv(tmp_path / 'cut.csv')
    assert len((tmp_path / 'cut.csv').read_text().splitlines()) == 2 + len(cut_1d)

def test_plot_cut_1d_shared_axes(dataset, cut_1d):
    ax = dataset.plot_cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8, fmt='o', label='first')
    start = ax.startPoint.copy()
    second = dataset.plot_cut_1d(HKL1, HKL2, 0.1, 0.02, 0.0, 0.8, ax=ax, label='second')
    assert second is ax and len(ax.lines) == 2
    assert np.array_equal(ax.startPoint, start) # Positions kept from the first cut
    assert [c.get_label() for c in ax.containers] == ['first', 'second']

def test_plot_cut_1d_background(dataset):
    dataset.generate_background_model()
    try:
        ax = dataset.plot_cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8, background_subtraction=True,
                                 plot_foreground=True, plot_background=True)
        data = ax.Data
        assert [c.get_label() for c in ax.containers] == ['_Data', 'Foreground', 'Background']
        assert np.allclose(ax.lines[0].get_ydata(), data['Int'], equal_nan=True)
        assert np.allclose(ax.lines[1].get_ydata(), data['Int_Fg'], equal_nan=True)
        assert np.allclose(ax.lines[2].get_ydata(), data['Int_Bg'], equal_nan=True)
        counts = dataset.plot_cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8, counts=True,
                                     background_subtraction=True)
        assert np.allclose(counts.lines[0].get_ydata(),
                           counts.Data['intensity_Fg'] - counts.Data['intensity_Bg'])
    finally:
        dataset.remove_background_model()

# Q-E cuts along a path of Q points

Q_POINTS = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
ENERGY_BINS = np.arange(-1.0, 0.81, 0.1)

@pytest.fixture(scope="module")
def qe_line(dataset):
    return dataset.cut_qe_line(Q_POINTS, ENERGY_BINS, width=0.1, min_pixel=0.02)

def test_cut_qe_line(dataset, qe_line):
    qxqy = np.array([dataset.convert_to_qxqy(q) for q in Q_POINTS])
    assert np.allclose(qe_line.bins['q_points'], qxqy)
    lengths = np.linalg.norm(np.diff(qxqy, axis=0), axis=1)
    assert np.allclose(qe_line.bins['offsets'], [0.0, lengths[0]])
    assert sorted(qe_line.data['qCut'].unique()) == [0, 1]
    # Each segment is the cut_qe between its two points
    second = dataset.cut_qe(Q_POINTS[1], Q_POINTS[2], 0.1, 0.02, energy_bins=ENERGY_BINS)
    segment = qe_line.data[qe_line.data['qCut'] == 1]
    assert np.array_equal(segment['Int'].to_numpy(), second.data['Int'].to_numpy(), equal_nan=True)
    assert np.allclose(segment['BinDistance'],
                       np.linalg.norm(segment[['Qx', 'Qy']].to_numpy() - qxqy[1], axis=1))

def test_cut_qe_line_per_segment_and_errors(dataset):
    result = dataset.cut_qe_line(Q_POINTS, [ENERGY_BINS, ENERGY_BINS[::2]], width=[0.1, 0.2],
                                 min_pixel=0.02)
    assert [len(edges) for edges in result.bins['energy']] == [len(ENERGY_BINS),
                                                               len(ENERGY_BINS[::2])]
    assert result.parameters['width'] == [0.1, 0.2]
    with pytest.raises(ValueError, match='2 segments'):
        dataset.cut_qe_line(Q_POINTS, ENERGY_BINS, width=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match='shape'):
        dataset.cut_qe_line(Q_POINTS[:1], ENERGY_BINS)
    with pytest.raises(ValueError, match=r'\(n, 2\)'):
        dataset.cut_qe_line(Q_POINTS, ENERGY_BINS, rlu=False) # H, K, L given as Qx, Qy

def test_plot_cut_qe_line(dataset, qe_line):
    ax = Plotting.plot_cut_qe_line(qe_line, sample=dataset.sample, colorbar=True)
    assert len(ax.pmeshs) == 2 and ax.type == 'QELine'
    # One colour scale for all segments
    values = np.concatenate([mesh.get_array().compressed() for mesh in ax.pmeshs])
    assert {mesh.get_clim() for mesh in ax.pmeshs} == {(values.min(), values.max())}
    # The second segment starts where the first ends
    first, second = (mesh.get_coordinates()[0, :, 0] for mesh in ax.pmeshs)
    assert second.min() > first.max() - 0.05
    assert np.allclose(ax.calculatePosition(ax.OffSets[1]).flatten(), Q_POINTS[1])
    assert np.allclose(ax.calculatePosition(0.0).flatten(), Q_POINTS[0])
    ax.get_figure().canvas.draw()
    assert len(ax.get_xticklabels()[0].get_text().split('\n')) == 3
    assert ax.get_xlabel() == '[H [RLU], K [RLU], L [RLU]]'

def test_plot_cut_qe_line_options(dataset, qe_line, tmp_path):
    ax = Plotting.plot_cut_qe_line(qe_line, sample=dataset.sample, vmin=0, vmax=1e-3,
                                   ticks=6, plot_separator=True, separator_color='r')
    assert {mesh.get_clim() for mesh in ax.pmeshs} == {(0, 1e-3)}
    assert len(ax.xaxis.get_ticklocs()) == 6
    assert np.isclose(ax.xaxis.get_ticklocs(), ax.OffSets[1]).any() # A tick at each Q point
    separators = ax.collections[-1]
    assert np.allclose(separators.get_segments()[0][:, 0], ax.OffSets[1])
    none = Plotting.plot_cut_qe_line(qe_line, sample=dataset.sample, ticks=0)
    assert len(none.xaxis.get_ticklocs()) == 0
    qxqy = Plotting.plot_cut_qe_line(qe_line, rlu=False)
    assert qxqy.get_xlabel() == '[Qx [1/AA], Qy [1/AA]]'
    assert np.allclose(qxqy.calculatePosition(qxqy.OffSets[1]).flatten(), qe_line.bins['q_points'][1])
    ax.to_csv(tmp_path / 'line.csv')
    assert len((tmp_path / 'line.csv').read_text().splitlines()) == 2 + len(qe_line)
    with pytest.raises(ValueError, match='3D'):
        Plotting.plot_cut_qe_line(qe_line, sample=dataset.sample,
                                  ax=plt.figure().add_subplot(projection='3d'))

def test_plot_cut_qe_line_rlu_false(dataset, qe_line):
    qxqy = np.array([dataset.convert_to_qxqy(q) for q in Q_POINTS])
    ax = dataset.plot_cut_qe_line(qxqy, ENERGY_BINS, width=0.1, min_pixel=0.02, rlu=False)
    reference = Plotting.plot_cut_qe_line(qe_line, sample=dataset.sample)
    for mesh, expected in zip(ax.pmeshs, reference.pmeshs): # Same points: one file orientation
        assert np.array_equal(np.ma.filled(mesh.get_array(), np.nan),
                              np.ma.filled(expected.get_array(), np.nan), equal_nan=True)

# Energy scans at constant Q

Q_POINT = np.array([0.0, 0.2, 0.0])

@pytest.fixture(scope="module")
def constant_q(dataset):
    return dataset.cut_constant_q(Q_POINT, width=0.1, min_pixel=0.05, energy_min=-0.8,
                                  energy_max=0.6)

def test_plot_cut_constant_q(dataset, constant_q):
    ax = Plotting.plot_cut_constant_q(constant_q, sample=dataset.sample)
    x, y = ax.lines[0].get_data()
    assert np.allclose(x, constant_q.data['Energy'])
    assert np.allclose(y, constant_q.data['Int'], equal_nan=True)
    assert np.allclose(ax.Data['BinDistance'], constant_q.data['Energy'])
    assert np.allclose(ax.startPoint, Q_POINT, atol=1e-9) and ax.type == 'ConstantQ'
    assert ax.get_xlabel() == '[$Q_h$ [RLU], $Q_k$ [RLU], $Q_l$ [RLU], E [meV]]'
    ax.get_figure().canvas.draw()
    label = ax.get_xticklabels()[1].get_text().split('\n')
    assert len(label) == 4 and [float(v) for v in label[:3]] == pytest.approx(Q_POINT, abs=0.01)
    assert ax.result is constant_q

def test_plot_cut_constant_q_options(dataset, constant_q, tmp_path):
    plain = Plotting.plot_cut_constant_q(constant_q, show_q=False, counts=True)
    assert plain.get_xlabel() == 'E [meV]' # No sample needed without the Q point
    assert np.allclose(plain.lines[0].get_ydata(), constant_q.data['intensity'])
    plain.get_figure().canvas.draw()
    assert '\n' not in plain.get_xticklabels()[1].get_text()
    qxqy = Plotting.plot_cut_constant_q(constant_q, rlu=False)
    assert np.allclose(qxqy.startPoint, constant_q.parameters['q'])
    qxqy.to_csv(tmp_path / 'cut.csv')
    assert len((tmp_path / 'cut.csv').read_text().splitlines()) == 2 + len(constant_q)
    with pytest.raises(AttributeError, match='not understood'):
        Plotting.plot_cut_constant_q(constant_q, sample=dataset.sample, counts='unknown')
    with pytest.raises(AttributeError, match='sample is needed'):
        Plotting.plot_cut_constant_q(constant_q)
    with pytest.raises(ValueError, match='cut_constant_q'):
        Plotting.plot_cut_constant_q(dataset.cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8),
                                     sample=dataset.sample)

def test_plot_cut_constant_q_shared_axes_and_background(dataset):
    options = dict(width=0.1, min_pixel=0.05, energy_min=-0.8, energy_max=0.6)
    ax = dataset.plot_cut_constant_q(Q_POINT, label='first', **options)
    second = dataset.plot_cut_constant_q([0.0, 0.3, 0.0], ax=ax, label='second', **options)
    assert second is ax and [c.get_label() for c in ax.containers] == ['first', 'second']
    assert np.allclose(ax.startPoint, Q_POINT, atol=1e-9) # The first cut's Q point is kept
    dataset.generate_background_model()
    try: # Asking for the curves turns background subtraction on, as the old plotCut1DE
        curves = dataset.plot_cut_constant_q(Q_POINT, plot_foreground=True,
                                             plot_background=True, **options)
        assert [c.get_label() for c in curves.containers] == ['_Data', 'Foreground', 'Background']
        assert np.allclose(curves.lines[1].get_ydata(), curves.Data['Int_Fg'], equal_nan=True)
        assert np.allclose(curves.lines[0].get_ydata(), curves.Data['Int'], equal_nan=True)
    finally:
        dataset.remove_background_model()
