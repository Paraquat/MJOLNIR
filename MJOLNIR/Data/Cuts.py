"""Cuts through scattering data, as pure functions.

Each function takes the data points as 1-D arrays (in Qx, Qy and energy, all in the same
frame) and returns a CutResult. They do not know about DataSet: a DataSet gathers the
unmasked, rotated points (DataSet.gather_q, DataSet.gather) and passes them in. Cut positions
q1, q2 are in Qx, Qy [1/AA]; with a sample, bin positions are also given in H, K, L.

With background (the background counts of each point, e.g. from a background model), it is
summed over each bin like the counts, and subtracted: see _subtract_background.

With intensity_error (the standard deviation of each point's intensity, e.g. of a difference
of data sets), the squared errors are summed per bin and Int_err = sqrt(sum of variances) *
BinCount / (Monitor * Normalization); without it, counting statistics (the variance of a
point is its counts).
"""
import warnings
import numpy as np
import numpy.typing as npt
import pandas as pd

from MJOLNIR import _tools
from MJOLNIR.Data.Coordinates import convert_to_hkl
from MJOLNIR.Data.CutResult import CutResult, COLUMNS

FloatArray = npt.NDArray[np.float64]

def _to_hkl(sample, qxqy: FloatArray) -> FloatArray:
    """H, K, L of Qx, Qy points of shape (n, 2), or NaN without a sample"""
    if sample is None:
        return np.full((len(qxqy), 3), np.nan)
    return convert_to_hkl(sample, qxqy)

BACKGROUND_COLUMNS = [COLUMNS[key] for key in ['BackgroundIntensity', 'ForegroundIntensity',
                                               'Background', 'Foreground', 'ForegroundError',
                                               'BackgroundError']]

def _subtract_background(data: pd.DataFrame, background_sum, variance=None) -> pd.DataFrame:
    """Add the background columns to the bins of a cut, from the background counts summed
    over each bin, as the old cut1D did:

        - intensity_Bg, intensity_Fg: Background and measured counts in the bin
        - Int_Bg, Int_Fg: Both normalised like Int, with their errors Int_Bg_err, Int_Fg_err
        - Int: Int_Fg - Int_Bg, with the error Int_err from both counts

    intensity keeps the measured counts. variance is the variance of the measured counts per
    bin; without it, the counts themselves (counting statistics)."""
    scale = lambda column: column*data[COLUMNS['binCount']]/(data[COLUMNS['norm']]*data[COLUMNS['mon']])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Empty bins divide by zero
        data[COLUMNS['BackgroundIntensity']] = np.asarray(background_sum, dtype=float).flatten()
        data[COLUMNS['ForegroundIntensity']] = data[COLUMNS['intensity']]
        foreground = data[COLUMNS['ForegroundIntensity']]
        background = data[COLUMNS['BackgroundIntensity']]
        data[COLUMNS['Background']] = scale(background)
        data[COLUMNS['Foreground']] = scale(foreground)
        data[COLUMNS['int']] = data[COLUMNS['Foreground']] - data[COLUMNS['Background']]
        variance = foreground if variance is None else np.asarray(variance, dtype=float)
        data[COLUMNS['ForegroundError']] = scale(np.sqrt(variance))
        data[COLUMNS['BackgroundError']] = scale(np.sqrt(background))
        data[COLUMNS['intError']] = scale(np.sqrt(variance + background))
    return data

def _directions(q1, q2) -> tuple[FloatArray, FloatArray, float]:
    """Unit vector along the cut, unit vector orthogonal to it, and the cut length"""
    direction = np.asarray(q2, dtype=float) - np.asarray(q1, dtype=float)
    length = np.linalg.norm(direction)
    if np.isclose(length, 0):
        raise AttributeError(f'Provided Q points are equal. Got {q1} and {q2}.')
    direction /= length
    return direction, np.array([direction[1], -direction[0]]), length

def cut_1d(qx, qy, energy, intensity, normalization, monitor, q1, q2, width: float,
           min_pixel: float, energy_min: float, energy_max: float, sample=None,
           extend: bool = False, constant_bins: bool = False, background=None,
           intensity_error=None) -> CutResult:
    """1D cut through Q from q1 to q2, of full width width, summed over energies between
    energy_min and energy_max (the old DataSet.cut1D).

    Bins along the cut are at least min_pixel long, and merged where points are sparser
    (constant_bins: all exactly min_pixel). Without extend, only points between q1 and q2
    (with a 5% margin) are used. With background, Int is background subtracted;
    intensity_error as in the module docstring.

    bins: 'positions' (bin edges as Qx, Qy, E, shape (n+1, 3)), 'orthogonal' (the two edges
    across the cut, shape (2, 2)) and 'energy' (energy_min, energy_max).
    """
    q1 = np.asarray(q1, dtype=float)
    parameters = dict(q1=q1, q2=np.asarray(q2, dtype=float), width=width, min_pixel=min_pixel,
                      energy_min=energy_min, energy_max=energy_max, extend=extend,
                      constant_bins=constant_bins)
    direction, orthogonal, length = _directions(q1, q2)
    projection = np.array([direction, orthogonal])
    energy_edges = np.array([energy_min, energy_max])
    orthogonal_edges = [-width/2.0, width/2.0]
    orthogonal_positions = np.outer(orthogonal_edges, orthogonal)

    def empty(positions):
        columns = [COLUMNS[key] for key in ['qx', 'qy', 'h', 'k', 'l', 'e', 'intensity',
                                            'intError', 'mon', 'norm', 'binCount', 'int']]
        if background is not None:
            columns += BACKGROUND_COLUMNS
        return CutResult(pd.DataFrame([], columns=columns),
                         dict(positions=positions, orthogonal=orthogonal_positions,
                              energy=energy_edges), parameters)

    inside_energy = np.logical_and(energy <= energy_max, energy >= energy_min)
    if not inside_energy.any():
        return empty(np.array([]))

    points = np.array([qx[inside_energy], qy[inside_energy]])
    projected = np.dot(projection, points - q1.reshape(2, 1))
    weights = [intensity[inside_energy], monitor[inside_energy], normalization[inside_energy]]
    if background is not None:
        weights.append(background[inside_energy])
    if intensity_error is not None:
        weights.append(np.asarray(intensity_error)[inside_energy]**2)
    if not extend: # Only points between q1 and q2
        if constant_bins:
            inside_q = np.logical_and(projected[0] > 0.0, projected[0] < length)
        else:
            inside_q = np.logical_and(projected[0] > -0.05, projected[0] < length*1.05)
        projected = projected[:, inside_q]
        weights = [weight[inside_q] for weight in weights]

    inside_width = np.logical_and(projected[1] < orthogonal_edges[1],
                                  projected[1] > orthogonal_edges[0])
    start, end = _tools.minMax(projected[0][inside_width])
    if constant_bins:
        length_edges = np.arange(start, end + 0.5*min_pixel, min_pixel)
    else:
        length_edges = np.array(_tools.binEdges(projected[0][inside_width], min_pixel,
                                                startPoint=start, endPoint=end))
    if len(length_edges) == 0:
        return empty(np.array([]))

    sums = [np.asarray(x).flatten() for x in _tools.histogramdd(
        projected.T, bins=[length_edges, orthogonal_edges],
        weights=[weight.flatten() for weight in weights], returnCounts=True)]
    counts, monitor_sum, normalization_sum = sums[:3]
    bin_count = sums[-1]
    variance = sums[-2] if intensity_error is not None else counts

    edges_qxqy = np.outer(length_edges, direction) + q1
    edges = np.concatenate([edges_qxqy, np.full((len(edges_qxqy), 1), 0.5*(energy_min + energy_max))],
                           axis=1)
    centres = 0.5*(edges[:-1] + edges[1:])
    hkl_edges = _to_hkl(sample, edges[:, :2])
    hkl = 0.5*(hkl_edges[:-1] + hkl_edges[1:])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Empty bins divide by zero
        denominator = normalization_sum*monitor_sum # Operation order as the old code
        data = pd.DataFrame({
            COLUMNS['qx']: centres[:, 0], COLUMNS['qy']: centres[:, 1],
            COLUMNS['h']: hkl[:, 0], COLUMNS['k']: hkl[:, 1], COLUMNS['l']: hkl[:, 2],
            COLUMNS['e']: centres[:, 2],
            COLUMNS['intensity']: counts.astype(float),
            COLUMNS['mon']: monitor_sum.astype(float),
            COLUMNS['norm']: normalization_sum.astype(float),
            COLUMNS['binCount']: bin_count.astype(int),
            COLUMNS['intError']: np.sqrt(variance)*bin_count/denominator,
            COLUMNS['int']: counts*bin_count/denominator,
        })
    if background is not None:
        data = _subtract_background(data, sums[3], variance)
    return CutResult(data, dict(positions=edges, orthogonal=orthogonal_positions,
                                energy=energy_edges), parameters)

def cut_constant_q(qx, qy, energy, intensity, normalization, monitor, q, width: float,
                   min_pixel: float, energy_min: float, energy_max: float, sample=None,
                   constant_bins: bool = False, background=None,
                   intensity_error=None) -> CutResult:
    """Energy scan at constant Q: the points within width/2 of q (a disc of diameter width,
    as cut_1d's width is the full width across its line) with energy_min <= E <= energy_max,
    binned in energy (the old DataSet.cut1DE, whose width was the radius).

    Energy bins are at least min_pixel wide, and merged where energies are sparser
    (constant_bins: all exactly min_pixel, from the lowest energy). With background, Int is
    background subtracted. intensity_error is the standard deviation of each point's
    intensity (e.g. of a difference of data sets); without it, counting statistics. Its
    squares are summed per bin, so Int_err = sqrt(sum of variances) * BinCount /
    (Monitor * Normalization).

    bins: 'energy' (bin edges), 'q' (the point, Qx, Qy).
    """
    q = np.asarray(q, dtype=float).flatten()
    if q.shape != (2,):
        raise ValueError(f'q must be a point in Qx, Qy; got {q.size} values (H, K, L are '
                         'converted by DataSet.cut_constant_q with rlu=True)')
    parameters = dict(q=q, width=width, min_pixel=min_pixel, energy_min=energy_min,
                      energy_max=energy_max, constant_bins=constant_bins)
    hkl = _to_hkl(sample, q.reshape(1, 2))[0]

    inside = np.logical_and.reduce([np.hypot(qx - q[0], qy - q[1]) < width/2.0,
                                    energy >= energy_min, energy <= energy_max])
    energies = energy[inside]
    if constant_bins and energies.size > 0:
        start, end = _tools.minMax(energies)
        energy_edges = np.arange(start, end + 0.5*min_pixel, min_pixel)
    else:
        energy_edges = np.array(_tools.binEdges(energies, tolerance=min_pixel))

    columns = [COLUMNS[key] for key in ['qx', 'qy', 'h', 'k', 'l', 'e', 'intensity', 'mon',
                                        'norm', 'binCount', 'int', 'intError']]
    if background is not None:
        columns += BACKGROUND_COLUMNS
    if len(energy_edges) < 2:
        return CutResult(pd.DataFrame([], columns=columns), dict(energy=energy_edges, q=q),
                         parameters)

    weights = [intensity[inside], monitor[inside], normalization[inside]]
    weights.append(np.asarray(intensity_error)[inside]**2 if intensity_error is not None
                   else intensity[inside]) # Variance of the counts
    if background is not None:
        weights.append(background[inside])
    sums = [np.asarray(x).flatten() for x in _tools.histogramdd(
        energies, bins=[energy_edges], weights=[weight.flatten() for weight in weights],
        returnCounts=True)]
    counts, monitor_sum, normalization_sum, variance = sums[:4]
    bin_count = sums[-1]

    centres = 0.5*(energy_edges[1:] + energy_edges[:-1])
    ones = np.ones_like(centres)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Empty bins divide by zero
        denominator = normalization_sum*monitor_sum # Operation order as the old code
        data = pd.DataFrame({
            COLUMNS['qx']: q[0]*ones, COLUMNS['qy']: q[1]*ones,
            COLUMNS['h']: hkl[0]*ones, COLUMNS['k']: hkl[1]*ones, COLUMNS['l']: hkl[2]*ones,
            COLUMNS['e']: centres,
            COLUMNS['intensity']: counts.astype(float),
            COLUMNS['mon']: monitor_sum.astype(float),
            COLUMNS['norm']: normalization_sum.astype(float),
            COLUMNS['binCount']: bin_count.astype(int),
            COLUMNS['int']: counts*bin_count/denominator,
            COLUMNS['intError']: np.sqrt(variance)*bin_count/denominator,
        })
    if background is not None:
        data = _subtract_background(data, sums[4], variance)
    return CutResult(data, dict(energy=energy_edges, q=q), parameters)

def cut_qe(qx, qy, energy, intensity, normalization, monitor, q1, q2, width: float,
           min_pixel: float, energy_bins=None, energy_min=None, energy_max=None,
           energy_step=None, sample=None, extend: bool = False, background=None,
           intensity_error=None) -> CutResult:
    """2D cut in Q and energy: Q along the line q1 to q2 (full width width, bins of
    min_pixel), energy in energy_bins (or from energy_min to energy_max in steps of
    energy_step). The old DataSet.cutQE.

    Without extend, only points between q1 and q2 are used. Points above energy_max are left
    out, even where the last energy bin extends beyond it. Rows are ordered by energy bin,
    then Q bin. As in the old cutQE, the Energy column holds the lower edge of each energy
    bin. With background, Int is background subtracted; intensity_error as in the module
    docstring.

    bins: 'distance' (edges along the cut, from q1) and 'energy' (energy edges);
    'orthogonal_extent' the Qx, Qy of the data's extent across the cut, at its two ends.
    """
    if energy_bins is None:
        if any(x is None for x in [energy_min, energy_max, energy_step]):
            raise AttributeError('When no energy_bins are given, energy_min, energy_max and '
                                 f'energy_step must be given. Received: energy_min={energy_min}, '
                                 f'energy_max={energy_max}, and energy_step={energy_step}')
        energy_bins = np.arange(energy_min, energy_max + energy_step, energy_step)
        # As in the old cutQE, points above energy_max are left out, even where the last bin
        # reaches beyond it
    else:
        energy_min, energy_max = np.asarray(energy_bins, dtype=float)[[0, -1]]
    energy_bins = np.asarray(energy_bins, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    parameters = dict(q1=q1, q2=np.asarray(q2, dtype=float), width=width, min_pixel=min_pixel,
                      energy_bins=energy_bins, extend=extend)

    direction, orthogonal, length = _directions(q1, q2)
    projection = np.array([direction, orthogonal])
    orthogonal_edges = [-width/2.0, width/2.0]
    projected = np.dot(projection, np.array([qx, qy]) - q1.reshape(2, 1))

    # Extent of the data across the cut, at the line through q1
    start = np.dot(orthogonal, q1)*orthogonal/np.dot(orthogonal, orthogonal)
    orthogonal_extent = np.array([orthogonal*np.min(projected[1]) + start,
                                  orthogonal*np.max(projected[1]) + start])

    inside_energy = np.logical_and(energy <= energy_max, energy >= energy_min)
    inside_width = np.logical_and(projected[1] < orthogonal_edges[1],
                                  projected[1] > orthogonal_edges[0])
    inside = np.logical_and(inside_width, inside_energy)
    if not extend:
        inside &= np.logical_and(projected[0] > 0.0, projected[0] < length)

    distance = projected[0][inside]
    if not extend: # From q1 to q2, one pixel further to bin the whole range
        distance_edges = np.arange(0, length + min_pixel, min_pixel)
    else:
        distance_edges = np.arange(np.min(distance), np.max(distance), min_pixel)

    weights = [intensity[inside].flatten(), monitor[inside].flatten(),
               normalization[inside].flatten()]
    if background is not None:
        weights.append(background[inside].flatten())
    if intensity_error is not None:
        weights.append(np.asarray(intensity_error)[inside].flatten()**2)
    sums = _tools.histogramdd(np.array([distance, energy[inside]]).T,
                              bins=[distance_edges, energy_bins], weights=weights,
                              returnCounts=True)
    counts, monitor_sum, normalization_sum = sums[:3]
    bin_count = sums[-1]
    variance = sums[-2] if intensity_error is not None else counts

    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Empty bins divide by zero
        normalised = np.divide(counts*bin_count, monitor_sum*normalization_sum)
        normalised_error = np.sqrt(variance)*bin_count.astype(float)/(monitor_sum*normalization_sum)

    centres = np.outer(distance_edges, direction) + q1
    centres = 0.5*(centres[:-1] + centres[1:]) # Qx, Qy of the Q bins
    hkl = _to_hkl(sample, centres)

    frames = []
    for column, lower_edge in enumerate(energy_bins[:-1]):
        frames.append(pd.DataFrame({
            COLUMNS['qx']: centres[:, 0], COLUMNS['qy']: centres[:, 1],
            COLUMNS['h']: hkl[:, 0], COLUMNS['k']: hkl[:, 1], COLUMNS['l']: hkl[:, 2],
            COLUMNS['e']: np.repeat(lower_edge, len(centres)),
            COLUMNS['int']: normalised[:, column].astype(float),
            COLUMNS['intError']: normalised_error[:, column].astype(float),
            COLUMNS['intensity']: counts[:, column].astype(float),
            COLUMNS['mon']: monitor_sum[:, column].astype(float),
            COLUMNS['norm']: normalization_sum[:, column].astype(float),
            COLUMNS['binCount']: bin_count[:, column].astype(int),
        }))
        if background is not None:
            frames[-1] = _subtract_background(frames[-1], sums[3][:, column],
                                              variance[:, column])
    return CutResult(pd.concat(frames), dict(distance=distance_edges, energy=energy_bins,
                                             orthogonal_extent=orthogonal_extent), parameters)

def _per_segment(value, n_segments: int, name: str) -> list:
    """One value per segment: a single value is repeated"""
    if not isinstance(value, (list, tuple, np.ndarray)):
        return [value]*n_segments
    if len(value) != n_segments:
        raise ValueError(f'{name} has {len(value)} values for {n_segments} segments')
    return list(value)

def cut_qe_line(qx, qy, energy, intensity, normalization, monitor, q_points, energy_bins,
                width=0.1, min_pixel=0.01, sample=None, background=None,
                intensity_error=None) -> CutResult:
    """Q-E cuts along a path of Q points (Qx, Qy, shape (n, 2), n >= 2): one cut_qe from each
    point to the next, without extend (the old DataSet.cutQELine). energy_bins, width and
    min_pixel are shared by all segments, or given per segment (a list of energy bin arrays,
    of widths, of pixel sizes).

    data holds the rows of all segments, with qCut the segment index and BinDistance the
    distance of the bin from the segment's start [1/AA]. bins: 'q_points', 'offsets' (the
    distance along the path where each segment starts), and per segment 'distance' and
    'energy' (lists of the cut_qe bins).
    """
    q_points = np.asarray(q_points, dtype=float)
    if q_points.ndim != 2 or q_points.shape[1] != 2 or len(q_points) < 2:
        raise ValueError(f'q_points must be at least 2 points in Qx, Qy, shape (n, 2); got '
                         f'shape {q_points.shape}')
    n_segments = len(q_points) - 1
    if np.ndim(energy_bins[0]) == 0: # One array of edges for all segments
        energy_bins = [np.asarray(energy_bins, dtype=float)]*n_segments
    else:
        energy_bins = [np.asarray(edges, dtype=float)
                       for edges in _per_segment(energy_bins, n_segments, 'energy_bins')]
    widths = _per_segment(width, n_segments, 'width')
    min_pixels = _per_segment(min_pixel, n_segments, 'min_pixel')

    frames, distance_bins, energy_edges = [], [], []
    for index in range(n_segments):
        segment = cut_qe(qx, qy, energy, intensity, normalization, monitor, q_points[index],
                         q_points[index + 1], widths[index], min_pixels[index],
                         energy_bins=energy_bins[index], sample=sample, background=background,
                         intensity_error=intensity_error)
        data = segment.data.copy()
        data['qCut'] = index
        data[COLUMNS['plotPosition']] = np.linalg.norm(
            data[[COLUMNS['qx'], COLUMNS['qy']]].to_numpy() - q_points[index], axis=1)
        frames.append(data)
        distance_bins.append(segment.bins['distance'])
        energy_edges.append(segment.bins['energy'])
    lengths = np.linalg.norm(np.diff(q_points, axis=0), axis=1)
    offsets = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    parameters = dict(q_points=q_points, width=widths, min_pixel=min_pixels,
                      energy_bins=energy_bins)
    return CutResult(pd.concat(frames, ignore_index=True),
                     dict(q_points=q_points, offsets=offsets, distance=distance_bins,
                          energy=energy_edges), parameters)

def cut_powder(qx, qy, energy, intensity, normalization, monitor, q_bins, energy_bins,
               background=None, intensity_error=None) -> CutResult:
    """Powder map: the points binned by the length of Q and by energy, whatever their
    direction (the old DataSet.cutPowder). q_bins and energy_bins are the bin edges [1/AA],
    [meV]; the last edge of each is included.

    data has one row per bin, also empty ones (Int NaN), ordered by |Q| bin, then energy bin:
    q and Energy are the bin centres. With background, Int is background subtracted, with the
    columns of the other cuts (see _subtract_background). intensity_error is the standard
    deviation of each point's intensity (default: counting statistics); its squares are
    summed per bin.

    bins: 'q' and 'energy' (the edges). result.grid(column) gives a column as an array of
    shape (|Q| bins, energy bins).
    """
    q_bins = np.asarray(q_bins, dtype=float)
    energy_bins = np.asarray(energy_bins, dtype=float)
    for name, edges in [('q_bins', q_bins), ('energy_bins', energy_bins)]:
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError(f'{name} must be at least 2 increasing bin edges')
    q_length = np.hypot(qx, qy)
    variance = intensity if intensity_error is None else np.asarray(intensity_error)**2

    def binned(weights=None):
        return np.histogram2d(q_length, energy, bins=(q_bins, energy_bins), weights=weights)[0]
    counts, monitor_sum, normalization_sum = binned(intensity), binned(monitor), binned(normalization)
    bin_count, variance_sum = binned(), binned(variance)

    centres_q, centres_e = np.meshgrid(0.5*(q_bins[1:] + q_bins[:-1]),
                                       0.5*(energy_bins[1:] + energy_bins[:-1]), indexing='ij')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Empty bins divide by zero
        denominator = monitor_sum*normalization_sum # Operation order as the old code
        data = pd.DataFrame({
            COLUMNS['qLength']: centres_q.flatten(), COLUMNS['e']: centres_e.flatten(),
            COLUMNS['intError']: np.divide(np.sqrt(variance_sum)*bin_count, denominator).flatten(),
            COLUMNS['int']: np.divide(counts*bin_count, denominator).flatten(),
            COLUMNS['intensity']: counts.flatten(),
            COLUMNS['norm']: normalization_sum.flatten(),
            COLUMNS['mon']: monitor_sum.flatten(),
            COLUMNS['binCount']: bin_count.flatten().astype(int),
        })
    if background is not None:
        data = _subtract_background(data, binned(background).flatten(), variance_sum.flatten())
    return CutResult(data, dict(q=q_bins, energy=energy_bins),
                     dict(q_bins=q_bins, energy_bins=energy_bins, grid=('q', 'energy')))

def cut_q_plane(qx, qy, energy, intensity, normalization, monitor, energy_bins=None,
                energy_min=None, energy_max=None, x_bin_tolerance: float = 0.05,
                y_bin_tolerance: float = 0.05, sample=None, background=None,
                intensity_error=None) -> CutResult:
    """Constant-energy planes: the points with energy_min < E <= energy_max (or within each
    pair of energy_bins), binned in Qx and Qy (the binning of the old DataSet.plotQPlane).

    Each energy range is binned on its own, from the smallest to the largest Qx (Qy) of its
    points in steps of x_bin_tolerance (y_bin_tolerance). Rows are ordered by energy range,
    then Qx bin, then Qy bin; Energy is the middle of the range. With background, Int is
    background subtracted; intensity_error as in the module docstring.

    bins: 'energy' (the energy edges), 'qx' and 'qy' (a list with the Qx, Qy edges of each
    energy range; ranges without points are left out).
    """
    if energy_bins is None:
        if energy_min is None or energy_max is None:
            raise AttributeError('Either energy_min and energy_max, or energy_bins, are to be '
                                 'given.')
        if energy_min >= energy_max:
            raise AttributeError(f'Provided limits are either wrong or the same. Received '
                                 f'energy_min={energy_min} and energy_max={energy_max}, '
                                 'expects energy_min < energy_max.')
        energy_bins = [energy_min, energy_max]
    energy_bins = np.asarray(energy_bins, dtype=float)
    if len(energy_bins) < 2:
        raise AttributeError(f'At least 2 energy bin edges are needed. Received {energy_bins}')
    parameters = dict(energy_bins=energy_bins, x_bin_tolerance=x_bin_tolerance,
                      y_bin_tolerance=y_bin_tolerance)

    frames, qx_edges, qy_edges = [], [], []
    for lower, upper in zip(energy_bins[:-1], energy_bins[1:]):
        inside = np.logical_and(energy > lower, energy <= upper)
        if not inside.any():
            continue
        x, y = qx[inside], qy[inside]
        x_edges = np.arange(np.min(x), np.max(x) + 0.999*x_bin_tolerance, x_bin_tolerance)
        y_edges = np.arange(np.min(y), np.max(y) + 0.999*y_bin_tolerance, y_bin_tolerance)
        histogram = lambda weights: np.histogram2d(x, y, bins=(x_edges, y_edges),
                                                   weights=weights)[0]
        counts = histogram(intensity[inside]).astype(intensity.dtype)
        monitor_sum = histogram(monitor[inside]).astype(monitor.dtype)
        normalization_sum = histogram(normalization[inside]).astype(normalization.dtype)
        bin_count = histogram(np.ones_like(intensity[inside])).astype(intensity.dtype)
        variance = (counts if intensity_error is None
                    else histogram(np.asarray(intensity_error)[inside]**2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Empty bins divide by zero
            normalised = np.divide(counts*bin_count, monitor_sum*normalization_sum)
            normalised_error = np.divide(np.sqrt(variance)*bin_count,
                                         normalization_sum*monitor_sum)

        x_mesh = np.outer(x_edges, np.ones_like(y_edges))
        y_mesh = np.outer(np.ones_like(x_edges), y_edges)
        corners = lambda mesh: 0.25*(mesh[:-1, :-1] + mesh[:-1, 1:] + mesh[1:, 1:] + mesh[1:, :-1])
        x_centres, y_centres = corners(x_mesh).flatten(), corners(y_mesh).flatten()
        hkl = _to_hkl(sample, np.array([x_centres, y_centres]).T)
        frames.append(pd.DataFrame({
            COLUMNS['qx']: x_centres, COLUMNS['qy']: y_centres,
            COLUMNS['h']: hkl[:, 0], COLUMNS['k']: hkl[:, 1], COLUMNS['l']: hkl[:, 2],
            COLUMNS['e']: np.full(len(x_centres), 0.5*(lower + upper)),
            COLUMNS['intensity']: counts.flatten(),
            COLUMNS['mon']: monitor_sum.flatten(),
            COLUMNS['norm']: normalization_sum.flatten(),
            COLUMNS['binCount']: bin_count.flatten(),
            COLUMNS['int']: normalised.flatten(),
            COLUMNS['intError']: normalised_error.flatten(),
        }))
        if background is not None:
            frames[-1] = _subtract_background(frames[-1], histogram(background[inside]).flatten(),
                                              variance.flatten())
        qx_edges.append(x_edges)
        qy_edges.append(y_edges)
    data = pd.concat(frames) if frames else pd.DataFrame()
    return CutResult(data, dict(energy=energy_bins, qx=qx_edges, qy=qy_edges), parameters)

def grid_edges(step: float, positions) -> FloatArray:
    """Bin edges covering positions, in steps close to step: the range of the positions
    divided evenly, with the first and last bin centred on the smallest and largest position
    (the old calculateBins and calculateGrid3D, along one axis)"""
    positions = np.asarray(positions)
    low, high = np.min(positions), np.max(positions)
    centres = np.linspace(low, high, np.round(np.abs(high - low)/step).astype(int) + 1)
    if len(centres) < 2:
        raise AttributeError(f'The positions span less than half a step of {step}: at least '
                             'two bins are needed along each axis')
    widths = np.diff(centres)
    edges = np.empty(len(centres) + 1)
    edges[:-1] = centres - 0.5*np.append(widths, widths[-1]) # Operation order as the old code
    edges[-1] = edges[-2] + widths[-1]
    return edges

def bin_3d(qx, qy, energy, intensity, normalization, monitor, dx: float = None,
           dy: float = None, dz: float = None, bins=None, sample=None,
           background=None, intensity_error=None) -> CutResult:
    """Bin the points in voxels of Qx, Qy and energy, about dx * dy * dz in size (the old
    DataSet.binData3D). The steps are adjusted to divide the range of the points evenly (see
    grid_edges); bins (the edges along Qx, Qy and energy) replaces them.

    One row per voxel, in the order of a C-ordered (Qx, Qy, energy) array: result.grid(column)
    gives a column in that shape. Qx, Qy, Energy (and H, K, L with a sample) are the centres
    of the voxels. With background, Int is background subtracted; intensity_error as in the
    module docstring.

    bins: 'qx', 'qy' and 'energy' (the edges along each axis).
    """
    if bins is None:
        if any(step is None for step in [dx, dy, dz]):
            raise AttributeError('Either dx, dy and dz, or bins, are to be given. Received: '
                                 f'dx={dx}, dy={dy}, dz={dz}')
        bins = [grid_edges(step, positions) for step, positions in zip([dx, dy, dz],
                                                                       [qx, qy, energy])]
    bins = [np.asarray(edges, dtype=float) for edges in bins]
    parameters = dict(dx=dx, dy=dy, dz=dz, grid=('qx', 'qy', 'energy'))

    weights = [np.asarray(intensity).flatten(), np.asarray(monitor).flatten(),
               np.asarray(normalization).flatten()]
    if background is not None:
        weights.append(np.asarray(background).flatten())
    if intensity_error is not None:
        weights.append(np.asarray(intensity_error).flatten()**2)
    sums = _tools.histogramdd(np.array([qx, qy, energy]).T, bins=bins, weights=weights,
                              returnCounts=True)
    counts, monitor_sum, normalization_sum = (np.asarray(x).flatten() for x in sums[:3])
    bin_count = np.asarray(sums[-1]).flatten()
    variance = np.asarray(sums[-2]).flatten() if intensity_error is not None else counts

    centres = [0.5*(edges[:-1] + edges[1:]) for edges in bins]
    x, y, z = (axis.flatten() for axis in np.meshgrid(*centres, indexing='ij'))
    hkl = _to_hkl(sample, np.array([x, y]).T)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Empty voxels divide by zero
        denominator = monitor_sum*normalization_sum
        data = pd.DataFrame({
            COLUMNS['qx']: x, COLUMNS['qy']: y,
            COLUMNS['h']: hkl[:, 0], COLUMNS['k']: hkl[:, 1], COLUMNS['l']: hkl[:, 2],
            COLUMNS['e']: z,
            COLUMNS['intensity']: counts.astype(float),
            COLUMNS['mon']: monitor_sum.astype(float),
            COLUMNS['norm']: normalization_sum.astype(float),
            COLUMNS['binCount']: bin_count.astype(int),
            COLUMNS['int']: np.divide(counts*bin_count, denominator),
            COLUMNS['intError']: np.divide(np.sqrt(variance)*bin_count, denominator),
        })
    if background is not None:
        data = _subtract_background(data, np.asarray(sums[3]).flatten(), variance)
    return CutResult(data, dict(qx=bins[0], qy=bins[1], energy=bins[2]), parameters)
