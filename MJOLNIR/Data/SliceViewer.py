"""Interactive slices through data binned in Qx, Qy and energy (Cuts.bin_3d), the library side
of the old DataSet.View3D / Viewer3D.

The viewer is plain matplotlib (any backend, also notebooks). It shows one slice at a time:
constant energy (axis 2, on reciprocal lattice axes with rlu), constant Qx (axis 0) or constant
Qy (axis 1). Cuts drawn on a slice are passed to one callback, on_cut(kind, parameters); without
one, a viewer made from a DataSet plots them itself. Applications such as MJOLNIR-GUI pass
their own on_cut to route the cuts into their windows.

Keys: up/right and down/left (or the scroll wheel) step through the slices; 0, 1, 2 choose the
slicing axis; c switches cutting on and off.
"""
import warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Slider

from MJOLNIR.Data import Plotting, RLUAxes
from MJOLNIR.Data.Coordinates import align_to_lattice, convert_to_hkl
from MJOLNIR.Data.CutResult import CutResult, COLUMNS

AXIS_NAMES = ('qx', 'qy', 'energy')
AXIS_LABELS = {'qx': r'$Q_x$ [$\AA^{-1}$]', 'qy': r'$Q_y$ [$\AA^{-1}$]', 'energy': 'E [meV]'}

class SliceViewer:
    """Slices through a CutResult of Cuts.bin_3d / DataSet.bin_3d.

    Args:

        - result (CutResult): Data binned on a Qx, Qy, energy grid

    Kwargs:

        - sample (Sample): The data set's sample, for H, K, L (needed for rlu) (default None)

        - rlu (bool): The result was binned aligned to the lattice (bin_3d(rlu=True)): constant
          energy slices on reciprocal lattice axes, positions in H, K, L (default True)

        - axis (int): Slicing axis: 0 Qx, 1 Qy, 2 energy (default 2)

        - counts (bool or str): Show the counts in each voxel (True), the sensitivity
          ('sensitivity': Normalization/BinCount) or the normalised intensity Int (False,
          default)

        - log (bool): Show log10 of the values (default False)

        - vmin, vmax (float): Colour scale (default: smallest and largest value)

        - cmap: Colour map (default 'viridis')

        - grid (bool): Grid on the Q-E slices (default False; lattice axes always have one)

        - adjustable (bool): Two sliders to set the colour scale (default True)

        - on_click (callable): Called with the read-out text when a slice is clicked outside
          cutting mode: the position and the voxel's Int, counts, normalisation, monitor and
          bin count (default print; None: no read-out). read_out(x, y) gives the same text

        - on_cut (callable): Called as on_cut(kind, parameters) for every cut drawn, kind one
          of '1d', 'qe', 'constant_q' and parameters the arguments of the DataSet method
          (cut_1d, cut_qe, cut_constant_q) (default None: dataset plots the cut)

        - dataset (DataSet): Data set the result was binned from, for the default cuts
          (default None)

        - cut_width (float): Width of drawn cuts [1/AA] (default 0.1)

        - cut_min_pixel (float): Bin size along drawn 1D and Q-E cuts [1/AA] (default 0.02)

        - spurion_peaks (list): Bragg peaks (H, K, L) whose Currat-Axe spurions are marked
          (default None)

        - Ei (list): Incoming energies of the data, for the spurions (default None)

        - ef_limits ((float, float)): Outgoing energies the instrument covers, for the
          spurions (default None: all)

    The slider below the plot and the keys move through the slices (see the module help).
    """

    def __init__(self, result: CutResult, sample=None, rlu: bool = True, axis: int = 2,
                 counts=False, log: bool = False, vmin=None, vmax=None, cmap='viridis',
                 grid: bool = False, adjustable: bool = True, on_click=print, on_cut=None,
                 dataset=None,
                 cut_width: float = 0.1, cut_min_pixel: float = 0.02, spurion_peaks=None,
                 Ei=None, ef_limits=None):
        if result.parameters.get('grid') != ('qx', 'qy', 'energy'):
            raise ValueError('SliceViewer needs a result binned on a Qx, Qy, energy grid '
                             '(Cuts.bin_3d)')
        if rlu and sample is None:
            raise AttributeError('A sample is needed for reciprocal lattice axes (rlu=True)')
        self.result = result
        self.rlu = rlu
        self.sample = sample
        # The frame the points were binned in: aligned to the lattice with rlu
        self.frame_sample = align_to_lattice(sample) if (rlu and sample is not None) else sample
        self.edges = [np.asarray(result.bins[name], dtype=float) for name in AXIS_NAMES]
        self.centres = [0.5*(e[1:] + e[:-1]) for e in self.edges]
        self.values = self._values(counts, log)
        self.on_click = on_click
        self.on_cut = on_cut
        self.dataset = dataset
        self.cut_width, self.cut_min_pixel = cut_width, cut_min_pixel
        self.cuts = [] # What on_cut returned for each drawn cut
        self.cutting = False
        self._cut_start = None
        self._cut_line = None

        finite = self.values.compressed()
        vmin = (finite.min() if finite.size else 0.0) if vmin is None else vmin
        vmax = (finite.max() if finite.size else 1.0) if vmax is None else vmax

        self.cmap = cmap
        self._clim = (vmin, vmax)
        self.mesh = None
        self.axis = None # Set by set_axis below; widgets may draw before
        self.index = 0
        self.slider = None

        self.figure = plt.figure(figsize=(8, 6))
        position = [0.12, 0.2, 0.62, 0.72]
        if rlu:
            # createQAxis reads the sample(s) of a data set: the first one sets the frame
            self.plane_ax = RLUAxes.createQAxis(SimpleNamespace(sample=[sample]), rlu=True,
                                                withoutOnClick=True, figure=self.figure)
        else:
            self.plane_ax = self.figure.add_subplot(111)
            self.plane_ax.set_aspect('equal')
            self.plane_ax.set_xlabel(AXIS_LABELS['qx'])
            self.plane_ax.set_ylabel(AXIS_LABELS['qy'])
        self.plane_ax.set_position(position)
        self.qe_ax = self.figure.add_axes(position)
        self.qe_ax.grid(grid)
        self.qe_ax.xaxis.set_major_formatter(FuncFormatter(self._qe_tick))
        for ax in (self.plane_ax, self.qe_ax):
            ax.format_coord = self._format_coord

        self.slider_ax = self.figure.add_axes([0.12, 0.06, 0.62, 0.03])
        self.clim_sliders = []
        if adjustable:
            low_ax = self.figure.add_axes([0.82, 0.2, 0.03, 0.72])
            high_ax = self.figure.add_axes([0.9, 0.2, 0.03, 0.72])
            low, high = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
            low, high = min(low, vmin), max(high, vmax) # The given scale is reachable
            self.clim_sliders = [Slider(low_ax, 'min', low, high, valinit=vmin,
                                        orientation='vertical'),
                                 Slider(high_ax, 'max', low, high, valinit=vmax,
                                        orientation='vertical')]
            for clim_slider in self.clim_sliders:
                clim_slider.on_changed(lambda _: self.set_clim(self.clim_sliders[0].val,
                                                                self.clim_sliders[1].val))

        self.spurions = self._spurions(spurion_peaks, Ei, ef_limits)
        self._spurion_lines = []
        self.set_axis(axis)

        canvas = self.figure.canvas
        self._connections = [
            canvas.mpl_connect('key_press_event', self._on_key),
            canvas.mpl_connect('scroll_event', self._on_scroll),
            canvas.mpl_connect('button_press_event', self._on_press),
            canvas.mpl_connect('button_release_event', self._on_release),
            canvas.mpl_connect('motion_notify_event', self._on_motion)]

    # Values

    def _values(self, counts, log: bool) -> np.ma.MaskedArray:
        grid = self.result.grid
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Empty voxels divide by zero
            if counts is True:
                values = grid(COLUMNS['intensity']).astype(float)
            elif counts is False:
                values = grid(COLUMNS['int']).astype(float)
            elif str(counts).lower() == 'sensitivity':
                values = grid(COLUMNS['norm'])/grid(COLUMNS['binCount'])
            else:
                raise AttributeError(f'Provided counts attribute not understood. Received "{counts}"')
            if log:
                values = np.log10(values + 1e-20)
        return np.ma.masked_invalid(values)

    @property
    def axes(self):
        """The axes showing the current slice"""
        return self.plane_ax if self.axis == 2 else self.qe_ax

    def slice_values(self, axis=None, index=None) -> np.ma.MaskedArray:
        """Values of a slice (default: the current one), indexed [x, y] of its plot"""
        axis = self.axis if axis is None else axis
        index = self.index if index is None else index
        return np.take(self.values, index, axis=axis)

    def _plot_edges(self, axis: int):
        """Edges along the plot's x and y for slices along axis"""
        return {2: (self.edges[0], self.edges[1]), 1: (self.edges[0], self.edges[2]),
                0: (self.edges[1], self.edges[2])}[axis]

    # Changing the slice

    def set_axis(self, axis: int):
        """Slice along axis: 0 Qx, 1 Qy, 2 energy. Keeps the middle slice"""
        if axis not in (0, 1, 2):
            raise AttributeError(f'Axis provided not recognized. Should be 0, 1, or 2 but got {axis}')
        self.axis = axis
        self.plane_ax.set_visible(axis == 2)
        self.qe_ax.set_visible(axis != 2)
        if self.mesh is not None:
            self.mesh.remove()
        x_edges, y_edges = self._plot_edges(axis)
        self.mesh = self.axes.pcolormesh(x_edges, y_edges, self.slice_values(axis, 0).T,
                                         cmap=self.cmap, zorder=10)
        self.mesh.set_clim(*self._clim)
        if axis != 2:
            self.qe_ax.set_xlabel(self._qe_xlabel())
            self.qe_ax.set_ylabel(AXIS_LABELS['energy'])
        self.axes.set_xlim(x_edges[0], x_edges[-1])
        self.axes.set_ylim(y_edges[0], y_edges[-1])

        self.slider_ax.clear()
        n_slices = len(self.centres[axis])
        self.slider = Slider(self.slider_ax, AXIS_NAMES[axis], 0, n_slices - 1,
                             valinit=n_slices//2, valstep=1)
        self.slider.on_changed(lambda value: self.set_slice(int(value)))
        self.set_slice(n_slices//2)

    def set_slice(self, index: int):
        """Show slice index along the current axis"""
        index = int(np.clip(index, 0, len(self.centres[self.axis]) - 1))
        self.index = index
        self.mesh.set_array(self.slice_values().T)
        if self.slider is not None and int(self.slider.val) != index:
            self.slider.set_val(index) # Calls set_slice again with the same index
        if self.slider is not None:
            self.slider.valtext.set_text(self.slice_label())
        self.axes.set_title(self.slice_label())
        self._draw_spurions()
        self.figure.canvas.draw_idle()

    def slice_label(self) -> str:
        """Position of the current slice, e.g. 'E = 1.250 meV'"""
        value = self.centres[self.axis][self.index]
        if self.axis == 2:
            return f'E = {value:.3f} meV'
        return f'{"Qx" if self.axis == 0 else "Qy"} = {value:.3f} 1/AA'

    def set_clim(self, vmin: float, vmax: float):
        self._clim = (vmin, vmax)
        self.mesh.set_clim(vmin, vmax)
        self.figure.canvas.draw_idle()

    def get_clim(self):
        return self._clim

    # Positions

    def position(self, x: float, y: float):
        """Qx, Qy and energy of a point (x, y) on the current slice"""
        value = self.centres[self.axis][self.index]
        if self.axis == 2:
            return x, y, value
        if self.axis == 1:
            return x, value, y
        return value, x, y

    def _hkl(self, qx, qy) -> np.ndarray:
        return convert_to_hkl(self.frame_sample, np.array([[qx, qy]], dtype=float))[0]

    def _qe_xlabel(self) -> str:
        if self.rlu:
            return '[H, K, L] [RLU]'
        return AXIS_LABELS['qy' if self.axis == 0 else 'qx']

    def _qe_tick(self, x, _) -> str:
        """Tick label on a Q-E slice: H, K, L of the position (rlu), else its Qx or Qy"""
        if not self.rlu or self.axis in (None, 2):
            return f'{x:.2f}'
        qx, qy, _ = self.position(x, 0.0)
        return '\n'.join(f'{value:.2f}' for value in self._hkl(qx, qy))

    def _format_coord(self, x, y) -> str: # pragma: no cover
        qx, qy, energy = self.position(x, y)
        position = (', '.join(f'{name} = {value:.3f}' for name, value in
                              zip('HKL', self._hkl(qx, qy)))
                    if self.rlu else f'Qx = {qx:.3f}, Qy = {qy:.3f}')
        value = self.value_at(qx, qy, energy)
        return f'{position}, E = {energy:.3f}, I = {value:.4e}'

    def voxel_at(self, qx: float, qy: float, energy: float) -> tuple[int, int, int] | None:
        """Index of the voxel containing a point, None outside the data"""
        index = []
        for edges, coordinate in zip(self.edges, (qx, qy, energy)):
            i = np.searchsorted(edges, coordinate) - 1
            if i < 0 or i >= len(edges) - 1:
                return None
            index.append(int(i))
        return tuple(index)

    def value_at(self, qx: float, qy: float, energy: float) -> float:
        """Shown value of the voxel containing a point, NaN outside the data"""
        index = self.voxel_at(qx, qy, energy)
        if index is None:
            return np.nan
        value = self.values[index]
        return np.nan if value is np.ma.masked else float(value)

    def read_out(self, x: float, y: float) -> str:
        """Position of a point (x, y) on the current slice and the values of its voxel (the
        old click print-out)"""
        text = self._format_coord(x, y)
        index = self.voxel_at(*self.position(x, y))
        if index is None:
            return text
        row = self.result.data.iloc[int(np.ravel_multi_index(index, self.values.shape))]
        return f'{text}, {Plotting.bin_text(row)}'

    # Currat-Axe spurions

    def _spurions(self, peaks, Ei, ef_limits):
        """Spurion positions per origin: array (n, 3) of Qx, Qy, energy in the binning frame"""
        if peaks is None or Ei is None or self.frame_sample is None:
            return {}
        energies = self.centres[2]
        spurions = {}
        for origin in ('Monochromator', 'Analyser'):
            points = []
            for incoming in np.unique(np.atleast_1d(Ei)):
                Ef = incoming - energies
                usable = Ef > 0
                if ef_limits is not None:
                    usable &= (Ef >= ef_limits[0]) & (Ef <= ef_limits[1])
                if not usable.any():
                    continue
                # (len(peaks), 1, len(Ef), 3): Qx, Qy, Qz in the frame of the sample
                qxqy = self.frame_sample.CurratAxe(
                    Ei=[incoming], Ef=Ef[usable], Bragg=peaks, spurionType=origin,
                    HKL=False)[:, 0, :, :2]
                energy = np.broadcast_to(energies[usable], qxqy.shape[:2])
                points.append(np.concatenate([qxqy, energy[..., None]], axis=-1).reshape(-1, 3))
            if points:
                spurions[origin] = np.concatenate(points)
        return spurions

    def _draw_spurions(self):
        for line in self._spurion_lines:
            line.remove()
        self._spurion_lines = []
        if not self.spurions:
            return
        width = np.diff(self.edges[self.axis])[self.index]
        centre = self.centres[self.axis][self.index]
        x_index, y_index = {2: (0, 1), 1: (0, 2), 0: (1, 2)}[self.axis]
        for origin, colour in (('Monochromator', 'r'), ('Analyser', 'k')):
            points = self.spurions.get(origin)
            if points is None:
                continue
            inside = np.abs(points[:, self.axis] - centre) <= width/2
            self._spurion_lines += self.axes.plot(
                points[inside, x_index], points[inside, y_index], linestyle='', marker='o',
                mfc='none', color=colour, zorder=20, label=f'Currat-Axe {origin.lower()}')

    # Cutting

    def _cut(self, kind: str, parameters: dict):
        """Pass a drawn cut on: to on_cut, or to the data set"""
        if self.on_cut is not None:
            returned = self.on_cut(kind, parameters)
        elif self.dataset is not None:
            returned = default_cut(self.dataset, kind, parameters)
        else:
            warnings.warn('Cut drawn, but the viewer has neither on_cut nor a data set')
            return None
        self.cuts.append(returned)
        return returned

    def _q(self, qx, qy):
        """A point as the DataSet methods take it: H, K, L with rlu, else Qx, Qy"""
        return self._hkl(qx, qy) if self.rlu else np.array([qx, qy])

    def cut_parameters(self, start, end) -> tuple[str, dict]:
        """Kind and DataSet arguments of a cut drawn from start to end (x, y on the current
        slice): a 1D cut on a constant-energy slice, a constant-Q cut for a click on a Q-E
        slice, a Q-E cut for a line on a Q-E slice"""
        (x1, y1), (x2, y2) = start, end
        if self.axis == 2:
            energy_edges = self.edges[2][self.index:self.index + 2]
            return '1d', dict(q1=self._q(x1, y1), q2=self._q(x2, y2), width=self.cut_width,
                              min_pixel=self.cut_min_pixel, energy_min=energy_edges[0],
                              energy_max=energy_edges[1], rlu=self.rlu)
        q_start, q_end = self.position(x1, y1)[:2], self.position(x2, y2)[:2]
        energy_step = np.diff(self.edges[2]).mean()
        if np.isclose(x1, x2, atol=1e-9) or np.hypot(*np.subtract(q_end, q_start)) < 1e-9:
            return 'constant_q', dict(q=self._q(*q_start), width=self.cut_width,
                                      min_pixel=energy_step, energy_min=self.edges[2][0],
                                      energy_max=self.edges[2][-1], rlu=self.rlu)
        low, high = sorted([y1, y2])
        energy_bins = self.edges[2][(self.edges[2] >= low) & (self.edges[2] <= high)]
        if len(energy_bins) < 2:
            energy_bins = self.edges[2][np.searchsorted(self.edges[2], low) - 1:][:2]
        return 'qe', dict(q1=self._q(*q_start), q2=self._q(*q_end), width=self.cut_width,
                          min_pixel=self.cut_min_pixel, energy_bins=energy_bins, rlu=self.rlu)

    def cut(self, start, end):
        """Make the cut drawn from start to end (x, y on the current slice), as a mouse drag
        does with cutting on"""
        return self._cut(*self.cut_parameters(start, end))

    # Events

    def _on_key(self, event): # pragma: no cover
        if event.key in ('up', 'right'):
            self.set_slice(self.index + 1)
        elif event.key in ('down', 'left'):
            self.set_slice(self.index - 1)
        elif event.key in ('0', '1', '2'):
            self.set_axis(int(event.key))
        elif event.key == 'c':
            self.cutting = not self.cutting

    def _on_scroll(self, event): # pragma: no cover
        if event.inaxes is self.axes:
            self.set_slice(self.index + (1 if event.button == 'up' else -1))

    def _on_press(self, event): # pragma: no cover
        toolbar = getattr(self.figure.canvas, 'toolbar', None)
        if (not self.cutting and self.on_click is not None and event.inaxes is self.axes
                and event.button == 1 and not (toolbar is not None and toolbar.mode)):
            self.on_click(self.read_out(event.xdata, event.ydata))
        if self.cutting and event.inaxes is self.axes and event.button == 1:
            self._cut_start = (event.xdata, event.ydata)
            self._cut_line, = self.axes.plot([event.xdata]*2, [event.ydata]*2, 'r-', zorder=30)

    def _on_motion(self, event): # pragma: no cover
        if self._cut_start is not None and event.inaxes is self.axes:
            self._cut_line.set_data([self._cut_start[0], event.xdata],
                                    [self._cut_start[1], event.ydata])
            self.figure.canvas.draw_idle()

    def _on_release(self, event): # pragma: no cover
        if self._cut_start is None:
            return
        start, self._cut_start = self._cut_start, None
        self._cut_line.remove()
        self._cut_line = None
        if event.inaxes is self.axes:
            self.cut(start, (event.xdata, event.ydata))
        self.figure.canvas.draw_idle()

    # Saving

    def to_csv(self, file_name):
        """Save the current slice: one row per voxel with its Qx, Qy, energy and value"""
        x_index, y_index = {2: (0, 1), 1: (0, 2), 0: (1, 2)}[self.axis]
        x, y = np.meshgrid(self.centres[x_index], self.centres[y_index], indexing='ij')
        columns = {AXIS_NAMES[x_index]: x.ravel(), AXIS_NAMES[y_index]: y.ravel(),
                   AXIS_NAMES[self.axis]: self.centres[self.axis][self.index],
                   'value': self.slice_values().filled(np.nan).ravel()}
        pd.DataFrame(columns).to_csv(file_name, index=False)

    def save_slices(self, folder, extension: str = 'png', **savefig_options) -> list[Path]:
        """Save every slice along the current axis as an image in folder; returns the files.
        The current slice is shown again afterwards"""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        current, files = self.index, []
        for index in range(len(self.centres[self.axis])):
            self.set_slice(index)
            files.append(folder / f'{AXIS_NAMES[self.axis]}_{index:03d}.{extension}')
            self.figure.savefig(files[-1], **savefig_options)
        self.set_slice(current)
        return files

def default_cut(dataset, kind: str, parameters: dict):
    """Plot a cut drawn in a SliceViewer with the DataSet's own methods; returns the axes"""
    if kind == '1d':
        return dataset.plot_cut_1d(**parameters)
    if kind == 'qe':
        return dataset.plot_cut_qe(**parameters)
    if kind == 'constant_q':
        return dataset.plot_cut_constant_q(**parameters)
    raise ValueError(f'Unknown cut kind "{kind}"')
