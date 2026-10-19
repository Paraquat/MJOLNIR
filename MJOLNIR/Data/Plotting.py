"""Plots of cut results. Each function draws a CutResult onto matplotlib axes (new axes if none
are given) and returns the axes; none of them computes a cut.

Not ported yet from the old DataSet plots: printing the data at a clicked point, and the
interactive modes of _interactiveSettings (cutting from a plot).
"""
import warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import MJOLNIR
from MJOLNIR import _tools
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

def _map_values(data, counts, log: bool, shape) -> np.ma.MaskedArray:
    """Values of a Q-E map in the shape of its bins: counts (True), the sensitivity
    ('sensitivity': Normalization/BinCount) or the normalised intensity (False); NaN masked;
    log10 with log"""
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
    return values

def plot_cut_1d(result: CutResult, sample=None, rlu: bool = True, ax=None, counts=False,
                show_energy: bool = True, plot_foreground: bool = False,
                plot_background: bool = False, **kwargs):
    """Plot a 1D cut (from Cuts.cut_1d) with error bars against the position along the cut
    (the old DataSet.plotCut1D).

    Args:

        - result (CutResult): The cut

    Kwargs:

        - sample (Sample): Sample of the cut's frame, needed for rlu (default None)

        - rlu (bool): Label the cut in H, K, L, else in Qx, Qy (default True)

        - ax (Axes): Axes to draw on (default None: new figure). Axes of an earlier 1D cut
          plot keep their positions, so several cuts can share them

        - counts (bool or str): Plot counts with sqrt(counts) errors (True), the sensitivity
          ('sensitivity': Normalization/BinCount) or the normalised intensity Int with Int_err
          (False, default). With background subtraction, counts are foreground minus
          background

        - show_energy (bool): Add the cut's mean energy to the tick labels (default True)

        - plot_foreground, plot_background (bool): With background subtraction, also plot the
          foreground and the background (default False)

    Other kwargs go to errorbar (default fmt '.'). The x axis runs from 0 at q1 to 1 at q2;
    the BinDistance column of ax.Data holds each bin's position on it. The returned axes have
    Data, result and to_csv.
    """
    if rlu and sample is None:
        raise AttributeError('A sample is needed to plot the cut in H, K, L (rlu=True)')
    if not (counts is True or counts is False or str(counts).lower() == 'sensitivity'):
        raise AttributeError(f'Provided counts attribute not understood. Received "{counts}"')
    if not hasattr(ax, 'calculatePositionInv'): # New axes, or axes not set up for a cut yet
        q1, q2 = result.parameters['q1'], result.parameters['q2']
        start, end = (_to_hkl(sample, q1), _to_hkl(sample, q2)) if rlu else (q1, q2)
        ax = _cut_axes(start, end, rlu=rlu, show_energy=show_energy, two_dimensional=False,
                       ax=ax)

    data = result.data
    if show_energy:
        ax.energy = np.mean(data[COLUMNS['e']])
    columns = [COLUMNS[key] for key in (['h', 'k', 'l'] if rlu else ['qx', 'qy'])]
    data[COLUMNS['plotPosition']] = ax.calculatePositionInv(np.asarray(data[columns])).flatten()
    column = lambda name: np.asarray(data[name], dtype=float)
    position = column(COLUMNS['plotPosition'])
    ax.set_ylabel('$I$ [arb.u.]')
    kwargs.setdefault('fmt', '.')
    kwargs.setdefault('label', '_Data')
    background = COLUMNS['BackgroundIntensity'] in data

    if counts is True:
        foreground = column(COLUMNS['intensity'])
        if background:
            values = foreground - column(COLUMNS['BackgroundIntensity'])
            errors = np.sqrt(foreground + column(COLUMNS['BackgroundIntensity']))
        else:
            values, errors = foreground, np.sqrt(foreground)
        ax.errorbar(position, values, yerr=errors, **kwargs)
        extra = {'Foreground': COLUMNS['ForegroundIntensity'],
                 'Background': COLUMNS['BackgroundIntensity']}
        extra_errors = {label: np.sqrt(column(name)) for label, name in extra.items()
                        if name in data}
    elif counts is False:
        ax.errorbar(position, column(COLUMNS['int']), yerr=column(COLUMNS['intError']), **kwargs)
        extra = {'Foreground': COLUMNS['Foreground'], 'Background': COLUMNS['Background']}
        if background:
            extra_errors = {'Foreground': column(COLUMNS['ForegroundError']),
                            'Background': column(COLUMNS['BackgroundError'])}
    else:
        ax.errorbar(position, column(COLUMNS['norm'])/column(COLUMNS['binCount']), **kwargs)
        extra = {}
    for label, wanted in [('Foreground', plot_foreground), ('Background', plot_background)]:
        if wanted and background and label in extra:
            ax.errorbar(position, column(extra[label]), yerr=extra_errors[label],
                        fmt=kwargs['fmt'], label=label)
    if len(position) > 0:
        ax.set_xlim(*_tools.minMax(position))
    ax.Data = data
    ax.result = result

    def to_csv(file_name):
        with open(file_name, 'w') as f:
            f.write(f'# CSV generated from MJOLNIR {MJOLNIR.__version__}. '
                    f'Shape of data is {(len(ax.Data),)}\n')
        ax.Data.fillna(-1).to_csv(file_name, mode='a')
    ax.to_csv = to_csv
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
    values = _map_values(data, counts, log, shape)

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

def plot_cut_qe_line(result: CutResult, sample=None, rlu: bool = True, ax=None, counts=False,
                     log: bool = False, vmin=None, vmax=None, cmap=None, colorbar: bool = False,
                     ticks: int | None = None, plot_separator: bool = False,
                     separator_color='k', separator_width=None, grid: bool = False, **kwargs):
    """Plot Q-E cuts along a path of Q points (from Cuts.cut_qe_line) side by side, as colour
    maps against the distance along the path [1/AA] (the old DataSet.plotCutQELine).

    Args:

        - result (CutResult): The cuts

    Kwargs:

        - sample (Sample): Sample of the cut's frame, needed for rlu (default None)

        - rlu (bool): Label positions in H, K, L, else in Qx, Qy (default True)

        - ax (Axes): Axes to draw on (default None: new figure); not 3D axes

        - counts, log, cmap, colorbar, grid: As plot_cut_qe

        - vmin, vmax (float): Colour scale of all segments (default: the minimum and maximum
          of all segments)

        - ticks (int): About this many ticks, at least one at each Q point; 0 for none
          (default None: matplotlib's)

        - plot_separator (bool): Vertical lines at the Q points between segments (default
          False), with separator_color (default 'k') and separator_width (line width)

    Other kwargs go to pcolormesh. The returned axes have set_clim, get_clim, to_csv, the Q
    points (QPoints in Qx, Qy; QPointsHKL) and OffSets (where each segment starts).
    """
    if rlu and sample is None:
        raise AttributeError('A sample is needed to plot the cut in H, K, L (rlu=True)')
    if ax is None:
        _, ax = plt.subplots()
    elif getattr(ax, 'name', None) == '3d':
        raise ValueError('plot_cut_qe_line draws on 2D axes, not 3D axes')
    q_points, offsets = result.bins['q_points'], result.bins['offsets']
    ax.QPoints, ax.OffSets, ax.rlu = q_points, offsets, rlu
    ax.QPointsHKL = _to_hkl(sample, q_points.T).T if sample is not None else None
    ax.fmtPrecisionString = '{:.3f}'

    def calculate_index(x):
        index = np.sum(offsets.reshape(-1, 1) < np.atleast_1d(x).reshape(1, -1), axis=0) - 1
        index[index == -1] = 0
        return index

    def calculate_position(x, rlu=None):
        """H, K, L (rlu) or Qx, Qy of positions x along the path, shape (3 or 2, len(x))"""
        rlu = ax.rlu if rlu is None else rlu
        x = np.atleast_1d(np.asarray(x, dtype=float))
        index = calculate_index(x)
        direction = q_points[index + 1] - q_points[index]
        direction /= np.linalg.norm(direction, axis=1).reshape(-1, 1)
        position = (x - offsets[index]).reshape(-1, 1)*direction + q_points[index]
        return _to_hkl(sample, position.T) if rlu else position.T

    ax.calculateIndex = calculate_index
    ax.calculatePosition = calculate_position

    data = result.data
    ax.Data = [segment for _, segment in data.groupby('qCut', sort=True)]
    ax.pmeshs = []
    for index, segment in enumerate(ax.Data):
        shape = (len(result.bins['energy'][index]) - 1, len(result.bins['distance'][index]) - 1)
        values = _map_values(segment, counts, log, shape)
        position = (segment[COLUMNS['plotPosition']].to_numpy() + offsets[index]).reshape(shape)
        energy = segment[COLUMNS['e']].to_numpy().reshape(shape)
        X, Y = np.meshgrid(position[0], energy[:, 0])
        ax.pmeshs.append(ax.pcolormesh(X, Y, values, shading='nearest', cmap=cmap, **kwargs))

    # One colour scale for all segments (the old plotCutQELine computed it but never set it)
    if vmin is None:
        vmin = np.min([mesh.get_array().min() for mesh in ax.pmeshs])
    if vmax is None:
        vmax = np.max([mesh.get_array().max() for mesh in ax.pmeshs])
    ax.set_clim = lambda vmin, vmax: [mesh.set_clim(vmin, vmax) for mesh in ax.pmeshs]
    ax.get_clim = lambda: ax.pmeshs[0].get_clim()
    ax.set_clim(vmin, vmax)

    if plot_separator:
        ax.vlines(offsets[1:], *ax.get_ylim(), color=separator_color, linewidth=separator_width)
    if colorbar:
        ax.colorbar = ax.get_figure().colorbar(ax.pmeshs[0])
    labels = ['H [RLU]', 'K [RLU]', 'L [RLU]'] if rlu else ['Qx [1/AA], Qy [1/AA]']
    ax.set_xlabel('[' + ', '.join(labels) + ']')
    ax.set_ylabel('E [meV]')

    if ticks == 0:
        ax.xaxis.set_ticks([])
    elif ticks is not None: # Spread over the segments by length, one at each Q point
        stops = np.asarray(list(offsets) + [np.round(ax.get_xlim(), 3)[1]])
        if ticks < len(stops):
            warnings.warn(f'Using {len(stops)} ticks instead of the wanted {ticks}')
            ticks = len(stops)
        wanted = np.diff(stops)*(ticks - len(stops))/stops[-1]
        optimal = np.floor(wanted).astype(int)
        optimal[np.argsort(wanted - optimal)[:ticks - optimal.sum() - len(stops)]] += 1
        positions = [np.linspace(start, stop, n + 1, endpoint=False)
                     for start, stop, n in zip(stops[:-1], stops[1:], optimal)]
        ax.xaxis.set_ticks(np.concatenate([*positions, [stops[-1]]]))

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '\n'.join(
        ax.fmtPrecisionString.format(p) for p in calculate_position(x).flatten())))

    def format_coord(x, y): # pragma: no cover
        names = ['H', 'K', 'L'] if ax.rlu else ['Qx', 'Qy']
        position = ', '.join(f'{name} = {ax.fmtPrecisionString.format(value)}'
                             for name, value in zip(names, calculate_position(x).flatten()))
        return f'{position}, E = {ax.fmtPrecisionString.format(y)}'
    ax.format_coord = format_coord

    ax.result = result
    ax.dE = np.diff(result.bins['energy'][0]).mean()
    ax.type = 'QELine'
    ax.grid(grid)

    def to_csv(file_name):
        with open(file_name, 'w') as f:
            f.write(f'# CSV generated from MJOLNIR {MJOLNIR.__version__}. '
                    f'{len(ax.Data)} segments (qCut)\n')
        data.fillna(-1).to_csv(file_name, mode='a')
    ax.to_csv = to_csv
    ax.get_figure().tight_layout()
    return ax

def plot_raw_1d(results: list[CutResult], ax=None, legend=True, grid: bool = True, **kwargs):
    """Plot raw counts (from DataSet.cut_raw_1d) against the first scan variable, one series
    per file (the old DataSet.plotRaw1D).

    Args:

        - results (list of CutResult): One raw cut per file

    Kwargs:

        - ax (Axes): Axes to draw on (default None: new figure)

        - legend (bool or list): Legend with the file names (True), the given names, or none
          (default True)

        - grid (bool): Show a grid (default True)

    Other kwargs go to scatter. The read-out under the cursor gives every scan variable at the
    position (interpolated for variables scanned together, e.g. A3 and A4) and the counts.
    The returned axes have results and to_csv.
    """
    if not results:
        raise ValueError('No raw cuts to plot')
    if ax is None:
        _, ax = plt.subplots()
    parameters = results[0].parameters['scan_parameters']
    units = results[0].parameters['scan_units']
    if not parameters:
        raise ValueError(f'"{results[0].parameters["file"]}" has no scan variables to plot against')
    if legend is True:
        legend = [result.parameters['file'] for result in results]
    elif legend and len(legend) != len(results):
        raise AttributeError(f'Provided list of legends does not match the number of files. '
                             f'Expected {len(results)}, but got {len(legend)}')
    for index, result in enumerate(results):
        label = legend[index] if legend else '_nolegend_'
        ax.scatter(result.data[parameters[0]], result.data[COLUMNS['intensity']], label=label,
                   **kwargs)
    xlabels = [f'{parameter} [{unit}]' for parameter, unit in zip(parameters, units)]
    ax.set_xlabel(xlabels[0])
    ax.set_ylabel('Int [counts]')
    if legend:
        ax.legend()
    ax.grid(grid)

    # All files' scan values, to give every variable at a position on the first one
    scan = np.concatenate([result.data[parameters].to_numpy(dtype=float).T for result in results],
                          axis=1)
    def format_coord(x, y): # pragma: no cover
        start, end = scan[0][0], scan[0][-1]
        fraction = (x - start)/(end - start) if end != start else 0.0
        values = [x] + [row[0] + fraction*(row[-1] - row[0]) for row in scan[1:]]
        steps = np.abs(np.diff(scan, axis=1)).mean(axis=1)
        decimals = [max(int(np.ceil(-np.log10(step))), 0) if step > 0 else 2 for step in steps]
        text = [f'{label} = {value:.{decimal}f}' for label, value, decimal
                in zip(xlabels, values, decimals)]
        return ', '.join(text + [f'Int [count] = {y:.0f}'])
    ax.format_coord = format_coord
    ax.results = results

    def to_csv(file_name):
        frames = [result.data.assign(file=result.parameters['file']) for result in results]
        pd.concat(frames, ignore_index=True).to_csv(file_name, index=False)
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

def plot_powder_background(model, ax=None):
    """Plot a powder-average background model (BackgroundModel.PowderBackground) as a colour
    map in |Q| and energy (the old plotPowderAverage). The axes have set_clim"""
    if ax is None:
        _, ax = plt.subplots()
    X, Y = np.meshgrid(model.q_bins, model.energy_bins)
    ax.p = ax.pcolormesh(X, Y, model.intensity.T)
    ax.set_xlabel('|Q| [1/A]')
    ax.set_ylabel('E [meV]')
    ax.get_figure().colorbar(ax.p)
    ax.set_clim = lambda vmin, vmax: ax.p.set_clim(vmin, vmax)
    return ax
