"""Export of cut results to other programs: ufit datasets (the old DataSet.generateUFitDataset)."""
import numpy as np

from MJOLNIR import _tools
from MJOLNIR.Data.CutResult import CutResult, COLUMNS

def to_ufit(result: CutResult, rlu: bool = True, meta: dict | None = None):
    """A ufit dataset (ufit.Dataset) of a 1D cut: Int with Int_err against the position along
    the cut (from cut_1d) or the energy (from cut_constant_q).

    Along Q, x runs from 0 at the first bin in units of the cut's direction, which the x label
    gives with the first bin's position as offsets, e.g. 'H-1.03, K-0.03, 0' and the mean energy
    (as the old generateUFitDataset). Along energy, x is the energy and the label gives the Q
    point.

    Args:

        - result (CutResult): A cut from cut_1d or cut_constant_q

    Kwargs:

        - rlu (bool): Positions in H, K, L (the result must have them, i.e. a sample), else in
          Qx, Qy (default True)

        - meta (dict): ufit metadata, e.g. instrument, experiment, title, datafilename
          (default None)
    """
    from ufit import Dataset # Imported here: ufit is only needed for this export
    data = result.data
    parameters = result.parameters
    along_q = 'q1' in parameters and 'energy_bins' not in parameters # cut_1d, not cut_qe
    if not along_q and 'q' not in parameters:
        raise ValueError('to_ufit takes 1D cuts: results of cut_1d or cut_constant_q')
    columns = [COLUMNS[key] for key in (['h', 'k', 'l'] if rlu else ['qx', 'qy'])]
    if len(data) == 0:
        raise ValueError('The cut is empty')
    if rlu and data[columns].isna().all().all():
        raise AttributeError('The cut has no H, K, L (made without a sample): use rlu=False')
    positions = data[columns].to_numpy(dtype=float)
    unit = '[RLU,meV]' if rlu else r'[$\AA^{-1}$,meV]'

    if along_q: # Along Q (cut_1d)
        q_rounding = max(int(-np.round(np.log10(parameters['min_pixel']))), 1)
        energy_range = parameters['energy_max'] - parameters['energy_min']
        e_rounding = max(int(np.round(6/np.abs(energy_range))), 1)
        if rlu: # LengthOrder only depends on the direction: the bins' span gives it in H, K, L
            direction = _tools.LengthOrder(positions[-1] - positions[0])
        else: # Norm2D depends on the length too: q2 - q1, as the old code
            direction = _tools.Norm2D(np.asarray(parameters['q2'], dtype=float)
                                      - np.asarray(parameters['q1'], dtype=float))
        # Signed distance from the first bin, in units of the direction
        x = np.dot(positions - positions[0], direction)/np.dot(direction, direction)
        labels = {} if rlu else dict(labels=[COLUMNS['qx'], COLUMNS['qy']])
        names = _tools.generateLabel(np.round(direction, q_rounding), **labels)[1:-1].split(', ')
        xlabel = []
        for name, offset in zip(names, np.round(positions[0], q_rounding)):
            if np.isclose(offset, 0):
                xlabel.append(name)
            elif name == '0':
                xlabel.append(f'{offset}')
            else:
                xlabel.append(name + f'{offset:+}')
        energy = 0.5*(parameters['energy_min'] + parameters['energy_max'])
        xcol = '\n'.join(xlabel) + '\n' + '{:.3}'.format(np.round(energy, e_rounding)) + '\n' + unit
    else: # Along energy (cut_constant_q)
        q_rounding = max(int(-np.round(np.log10(parameters['width']))), 1)
        x = data[COLUMNS['e']].to_numpy(dtype=float)
        point = ', '.join(str(value) for value in np.round(positions[0], q_rounding) + 0.0) # No -0.0
        xcol = f'E [meV] at ({point}) ' + ('[RLU]' if rlu else r'[$\AA^{-1}$]')

    values = np.array([x, data[COLUMNS['int']].to_numpy(dtype=float),
                       data[COLUMNS['intError']].to_numpy(dtype=float)]).T
    return Dataset(meta=dict(meta or {}), data=values, xcol=xcol, ycol=COLUMNS['intensity'],
                   name=COLUMNS['intensity'])
