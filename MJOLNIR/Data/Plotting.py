"""Plots of cut results. Each function draws a CutResult onto matplotlib axes (new axes if none
are given) and returns the axes; none of them computes a cut.

Not ported yet from the old DataSet plots: printing the data at a clicked point, and the
interactive modes of _interactiveSettings (cutting from a plot).
"""
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import MJOLNIR
from MJOLNIR.Data import RLUAxes
from MJOLNIR.Data.CutResult import CutResult, COLUMNS

def _to_hkl(sample, qxqy) -> np.ndarray:
    return np.array(sample.calculateQxQyToHKL(*np.asarray(qxqy, dtype=float)))

def _cut_axes(start, end, rlu: bool, show_energy: bool, two_dimensional: bool, ax=None):
    """Axes for a cut from start to end (H, K, L with rlu, else Qx, Qy): x runs from 0 at
    start to 1 at end, and its tick labels give the position along the cut (the old
    generate1DAxis)"""
    if ax is None:
        _, ax = plt.subplots()
    ax.startPoint = np.asarray(start, dtype=float)
    ax.endPoint = np.asarray(end, dtype=float)
    ax.plotDirection = (ax.endPoint - ax.startPoint).reshape(-1, 1)
    ax.rlu = rlu
    ax._2D = two_dimensional
    ax.energy = -1000.0 # Energy of a 1D cut, set when it is plotted
    ax._x_precision = 2
    ax.fmtPrecisionString = '{:.2f}'

    def calculate_position(x):
        return (x*ax.plotDirection.T + ax.startPoint).flatten()

    def calculate_position_inverse(points):
        points = np.asarray(points, dtype=float).reshape(-1, len(ax.startPoint))
        return (np.dot(points - ax.startPoint, ax.plotDirection)
                /np.dot(ax.plotDirection.T, ax.plotDirection))

    def set_precision(value):
        ax._x_precision = value
        ax.fmtPrecisionString = '{:.' + str(value) + 'f}'
        ax.get_figure().tight_layout()

    def calculate_x_precision(ax):
        # Enough decimals to tell the tick labels apart along the cut
        span = np.diff(ax.get_xlim())[0]*ax.plotDirection
        change = np.max(np.abs(span[ax.plotDirection.T.flatten() != 0]))/len(ax.xaxis.get_ticklocs())
        ax.set_precision(int(-np.floor(np.log10(change))) + 1)

    ax.calculatePosition = calculate_position
    ax.calculatePositionInv = calculate_position_inverse
    ax.set_precision = set_precision
    ax.calculateXPrecision = calculate_x_precision
    ax.calculateXPrecision(ax)

    if two_dimensional:
        ax.set_clim = lambda vmin, vmax: [mesh.set_clim(vmin, vmax) for mesh in ax.pmeshs]
        ax.get_clim = lambda: ax.pmeshs[0].get_clim()

    names = ['H', 'K', 'L'] if rlu else ['qx', 'qy']
    units = r'[$Q_h$ [RLU], $Q_k$ [RLU], $Q_l$ [RLU]' if rlu else r'[$Q_x [\AA^{-1}]$, $Q_y [\AA^{-1}]$'
    ax.set_xlabel(units + (', E [meV]]' if show_energy else ']'))

    def format_coord(x, y): # pragma: no cover
        position = ', '.join(f'{name} = {value:.3e}'
                             for name, value in zip(names, calculate_position(x)))
        if two_dimensional:
            return f'{position}, E = {y:.3f}'
        if show_energy:
            return f'{position}, E = {ax.energy:.3f}, I = {y:0.4e}'
        return f'{position}, I = {y:0.4e}'
    ax.format_coord = format_coord

    def tick_label(x, _):
        positions = list(calculate_position(x)) + ([ax.energy] if show_energy else [])
        return '\n'.join(ax.fmtPrecisionString.format(p) for p in positions)
    ax.xaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.callbacks.connect('xlim_changed', ax.calculateXPrecision)
    ax.get_figure().tight_layout()
    return ax

def plot_cut_qe(result: CutResult, sample=None, rlu: bool = True, ax=None, counts=False,
                log: bool = False, vmin=None, vmax=None, cmap=None, colorbar: bool = False,
                grid: bool = False, **kwargs):
    """Plot a Q-E cut (from Cuts.cut_qe) as a colour map: position along the cut against
    energy (the old DataSet.plotCutQE).

    Args:

        - result (CutResult): The cut

    Kwargs:

        - sample (Sample): Sample of the cut's frame, needed for rlu (default None)

        - rlu (bool): Label the cut in H, K, L, else in Qx, Qy (default True)

        - ax (Axes): Axes to draw on (default None: new figure)

        - counts (bool or str): Plot counts (True), the sensitivity ('sensitivity':
          Normalization/BinCount) or the normalised intensity (False, default)

        - log (bool): Plot log10 of the values (default False)

        - vmin, vmax, cmap: Colour scale (default None)

        - colorbar (bool): Add a colour bar (default False)

        - grid (bool): Show a grid (default False)

    Other kwargs go to pcolormesh. The returned axes have set_clim, get_clim and to_csv. As in
    the old plotCutQE, each energy row is drawn centred on the Energy column, the lower edge
    of its bin.
    """
    if rlu and sample is None:
        raise AttributeError('A sample is needed to plot the cut in H, K, L (rlu=True)')
    q1, q2 = result.parameters['q1'], result.parameters['q2']
    start, end = (_to_hkl(sample, q1), _to_hkl(sample, q2)) if rlu else (q1, q2)
    ax = _cut_axes(start, end, rlu=rlu, show_energy=False, two_dimensional=True, ax=ax)

    data = result.data
    distance_bins, energy_bins = len(result.bins['distance']) - 1, len(result.bins['energy']) - 1
    shape = (energy_bins, distance_bins)
    if counts is True:
        values = data[COLUMNS['intensity']].values
    elif counts is False:
        values = data[COLUMNS['int']].values
    elif str(counts).lower() == 'sensitivity':
        values = data[COLUMNS['norm']].values/data[COLUMNS['binCount']].values
    else:
        raise AttributeError(f'Provided counts attribute not understood. Received "{counts}"')
    values = np.ma.array(np.asarray(values, dtype=float).reshape(shape))
    values.mask = np.isnan(values)
    if log:
        values = np.log10(values + 1e-20)

    columns = [COLUMNS[key] for key in (['h', 'k', 'l'] if rlu else ['qx', 'qy'])]
    position = ax.calculatePositionInv(np.asarray(data[columns])).reshape(shape)
    data[COLUMNS['plotPosition']] = position.flatten()
    energy = np.asarray(data[COLUMNS['e']]).reshape(shape)

    ax.X, ax.Y = position[0], energy[:, 0]
    X, Y = np.meshgrid(ax.X, ax.Y)
    ax.pmeshs = [ax.pcolormesh(X, Y, values, shading='nearest', vmin=vmin, vmax=vmax,
                               cmap=cmap, **kwargs)]
    ax.set_ylabel('E [meV]')
    ax.Data = data
    ax.result = result
    ax.dE = np.diff(result.bins['energy']).mean()
    ax.width = result.parameters['width']
    ax.minPixel = result.parameters['min_pixel']
    ax.type = 'QE'
    if colorbar:
        ax.colorbar = ax.get_figure().colorbar(ax.pmeshs[0])
    ax.grid(grid)

    def to_csv(file_name):
        with open(file_name, 'w') as f:
            f.write(f'# CSV generated from MJOLNIR {MJOLNIR.__version__}. '
                    f'Shape of data is {(len(ax.X), len(ax.Y))}\n')
        ax.Data.fillna(-1).to_csv(file_name, mode='a')
    ax.to_csv = to_csv
    return ax

def plot_q_plane(result: CutResult, sample=None, rlu: bool = True, ax=None, log: bool = False,
                 vmin=None, vmax=None, cmap=None, colorbar: bool = False, zorder: int = 10):
    """Plot constant-energy planes (from Cuts.cut_q_plane) as colour maps in Qx, Qy, on axes
    with a reciprocal lattice grid (the old DataSet.plotQPlane).

    Args:

        - result (CutResult): The planes

    Kwargs:

        - sample (Sample): Sample of the data set's frame, for the RLU grid (default None)

        - rlu (bool): Draw on reciprocal lattice axes; the result must then be made from points
          aligned to the lattice (DataSet.gather_q(align_to_lattice=True)) (default True)

        - ax (Axes): Axes to draw on (default None: new axes from RLUAxes.createQAxis)

        - log (bool): Plot log10 of the intensity (default False)

        - vmin, vmax: Colour scale (default: smallest and largest value)

        - cmap: Colour map (default None)

        - colorbar (bool): Add a colour bar (default False)

        - zorder (int): Drawing order of the maps (default 10)

    The returned axes have set_clim, get_clim and to_csv.
    """
    if ax is None:
        if rlu and sample is None:
            raise AttributeError('A sample is needed for reciprocal lattice axes (rlu=True)')
        # createQAxis reads the sample(s) of a data set: the first one sets the frame
        ax = RLUAxes.createQAxis(SimpleNamespace(sample=[sample]), rlu=rlu, withoutOnClick=True)

    data = result.data
    ax.Qx, ax.Qy, ax.Int = [], [], []
    start = 0
    for x_edges, y_edges in zip(result.bins['qx'], result.bins['qy']):
        shape = (len(x_edges) - 1, len(y_edges) - 1)
        values = np.asarray(data[COLUMNS['int']].values[start:start + np.prod(shape)], dtype=float)
        start += np.prod(shape)
        ax.Qx.append(np.outer(x_edges, np.ones_like(y_edges)))
        ax.Qy.append(np.outer(np.ones_like(x_edges), y_edges))
        ax.Int.append(values.reshape(shape))
    if log:
        ax.Int = [np.log10(1e-20 + values) for values in ax.Int]
    if vmax is None:
        vmax = np.max([np.nanmax(values) for values in ax.Int])
    if vmin is None:
        vmin = np.min([np.nanmin(values) for values in ax.Int])

    pmeshs = []
    for Qx, Qy, values in zip(ax.Qx, ax.Qy, ax.Int):
        ax.grid(False)
        pmeshs.append(ax.pcolormesh(Qx, Qy, values, zorder=zorder, cmap=cmap))
    ax.set_aspect('equal')
    ax.grid(True, zorder=0)
    ax.pmeshs = list(getattr(ax, 'pmeshs', [])) + pmeshs
    ax.set_clim = lambda vmin, vmax: [mesh.set_clim(vmin, vmax) for mesh in ax.pmeshs]
    ax.get_clim = lambda: ax.pmeshs[0].get_clim()
    if colorbar:
        ax.colorbar = ax.get_figure().colorbar(ax.pmeshs[0], pad=0.1)
        ax.colorbar.set_label(log*'log(' + '$I$' + log*')' + '[arb.u.]', rotation=270)
    ax.set_clim(vmin, vmax)
    if ax.Qx:
        ax.set_xlim(np.min([Qx.min() for Qx in ax.Qx]), np.max([Qx.max() for Qx in ax.Qx]))
        ax.set_ylim(np.min([Qy.min() for Qy in ax.Qy]), np.max([Qy.max() for Qy in ax.Qy]))

    ax.result = result
    ax.Data = data
    ax.type = 'QPlane'
    energy_bins = result.bins['energy']
    ax.EMin, ax.EMax = energy_bins[0], energy_bins[-1]

    def to_csv(file_name):
        x_bins = len(result.bins['qx'][0]) - 1 if result.bins['qx'] else 0
        with open(file_name, 'w') as f:
            f.write(f'# CSV generated from MJOLNIR {MJOLNIR.__version__}. Shape of data is '
                    f'{(x_bins, len(data)//max(x_bins, 1))}\n')
        data.fillna(-1).to_csv(file_name, mode='a')
    ax.to_csv = to_csv
    return ax
