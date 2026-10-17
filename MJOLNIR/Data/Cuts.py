"""Cuts through scattering data, as pure functions.

Each function takes the data points as 1-D arrays (in Qx, Qy and energy, all in the same
frame) and returns a CutResult. They do not know about DataSet: a DataSet gathers the
unmasked, rotated points (DataSet.gather_q, DataSet.gather) and passes them in. Cut positions
q1, q2 are in Qx, Qy [1/AA]; with a sample, bin positions are also given in H, K, L.
"""
import warnings
import numpy as np
import numpy.typing as npt
import pandas as pd

from MJOLNIR import _tools
from MJOLNIR.Data.CutResult import CutResult, COLUMNS

FloatArray = npt.NDArray[np.float64]

def _to_hkl(sample, qxqy: FloatArray) -> FloatArray:
    """H, K, L of Qx, Qy points of shape (n, 2), or NaN without a sample"""
    if sample is None:
        return np.full((len(qxqy), 3), np.nan)
    return np.array(sample.calculateQxQyToHKL(qxqy[:, 0], qxqy[:, 1])).T

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
           extend: bool = False, constant_bins: bool = False) -> CutResult:
    """1D cut through Q from q1 to q2, of full width width, summed over energies between
    energy_min and energy_max (the old DataSet.cut1D).

    Bins along the cut are at least min_pixel long, and merged where points are sparser
    (constant_bins: all exactly min_pixel). Without extend, only points between q1 and q2
    (with a 5% margin) are used.

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
        return CutResult(pd.DataFrame([], columns=columns),
                         dict(positions=positions, orthogonal=orthogonal_positions,
                              energy=energy_edges), parameters)

    inside_energy = np.logical_and(energy <= energy_max, energy >= energy_min)
    if not inside_energy.any():
        return empty(np.array([]))

    points = np.array([qx[inside_energy], qy[inside_energy]])
    projected = np.dot(projection, points - q1.reshape(2, 1))
    weights = [intensity[inside_energy], monitor[inside_energy], normalization[inside_energy]]
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

    counts, monitor_sum, normalization_sum, bin_count = _tools.histogramdd(
        projected.T, bins=[length_edges, orthogonal_edges],
        weights=[weight.flatten() for weight in weights], returnCounts=True)
    counts, monitor_sum, normalization_sum, bin_count = (
        np.asarray(x).flatten() for x in [counts, monitor_sum, normalization_sum, bin_count])

    edges_qxqy = np.outer(length_edges, direction) + q1
    edges = np.concatenate([edges_qxqy, np.full((len(edges_qxqy), 1), 0.5*(energy_min + energy_max))],
                           axis=1)
    centres = 0.5*(edges[:-1] + edges[1:])
    hkl = _to_hkl(sample, centres[:, :2])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Empty bins divide by zero
        scale = bin_count/(normalization_sum*monitor_sum)
        data = pd.DataFrame({
            COLUMNS['qx']: centres[:, 0], COLUMNS['qy']: centres[:, 1],
            COLUMNS['h']: hkl[:, 0], COLUMNS['k']: hkl[:, 1], COLUMNS['l']: hkl[:, 2],
            COLUMNS['e']: centres[:, 2],
            COLUMNS['intensity']: counts.astype(float),
            COLUMNS['mon']: monitor_sum.astype(float),
            COLUMNS['norm']: normalization_sum.astype(float),
            COLUMNS['binCount']: bin_count.astype(int),
            COLUMNS['intError']: np.sqrt(counts)*scale,
            COLUMNS['int']: counts*scale,
        })
    return CutResult(data, dict(positions=edges, orthogonal=orthogonal_positions,
                                energy=energy_edges), parameters)

def cut_qe(qx, qy, energy, intensity, normalization, monitor, q1, q2, width: float,
           min_pixel: float, energy_bins=None, energy_min=None, energy_max=None,
           energy_step=None, sample=None, extend: bool = False) -> CutResult:
    """2D cut in Q and energy: Q along the line q1 to q2 (full width width, bins of
    min_pixel), energy in energy_bins (or from energy_min to energy_max in steps of
    energy_step). The old DataSet.cutQE.

    Without extend, only points between q1 and q2 are used. Rows are ordered by energy bin,
    then Q bin. As in the old cutQE, the Energy column holds the lower edge of each energy
    bin.

    bins: 'distance' (edges along the cut, from q1) and 'energy' (energy edges);
    'orthogonal_extent' the Qx, Qy of the data's extent across the cut, at its two ends.
    """
    if energy_bins is None:
        if any(x is None for x in [energy_min, energy_max, energy_step]):
            raise AttributeError('When no energy_bins are given, energy_min, energy_max and '
                                 f'energy_step must be given. Received: energy_min={energy_min}, '
                                 f'energy_max={energy_max}, and energy_step={energy_step}')
        energy_bins = np.arange(energy_min, energy_max + energy_step, energy_step)
    energy_bins = np.asarray(energy_bins, dtype=float)
    energy_min, energy_max = energy_bins[[0, -1]]
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

    counts, monitor_sum, normalization_sum, bin_count = _tools.histogramdd(
        np.array([distance, energy[inside]]).T, bins=[distance_edges, energy_bins],
        weights=[intensity[inside].flatten(), monitor[inside].flatten(),
                 normalization[inside].flatten()], returnCounts=True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # Empty bins divide by zero
        scale = bin_count/(monitor_sum*normalization_sum)
        normalised = counts*scale
        normalised_error = np.sqrt(counts)*bin_count.astype(float)/(monitor_sum*normalization_sum)

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
    return CutResult(pd.concat(frames), dict(distance=distance_edges, energy=energy_bins,
                                             orthogonal_extent=orthogonal_extent), parameters)
