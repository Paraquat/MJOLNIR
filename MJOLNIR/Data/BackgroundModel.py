"""Background models, fitted from arrays of data points. Like Cuts, they do not know about
DataSet: DataSet.generate_background_model gathers the points, fits a model, and stores the
background of every data point in its readers.

The old model, written for the original DataSet, is in BackgroundModel_old.py.
"""
import numpy as np
import numpy.typing as npt
import warnings

from MJOLNIR import _tools

FloatArray = npt.NDArray[np.float64]

class PowderBackground:
    """Background estimated as a powder average.

    Background from the sample environment, the sample holder and incoherent scattering is
    taken to be independent of the crystal's orientation: a function of |Q| and energy only.
    The model bins data points without signal (the signal masked out) in |Q| and energy, and
    normalises each bin like a cut: counts * points / (monitor * normalization).
    """

    def __init__(self, dQ: float = 0.025, dE: float = 0.05):
        """Kwargs:

            - dQ (float): Bin size along |Q| [1/AA] (default 0.025)

            - dE (float): Bin size along energy [meV] (default 0.05)
        """
        self.dQ = dQ
        self.dE = dE
        self.q_bins = None # Bin edges along |Q| and energy, set by fit
        self.energy_bins = None

    @property
    def dQ(self) -> float:
        return self._dQ

    @dQ.setter
    def dQ(self, value: float):
        if not value > 0:
            raise AttributeError(f'Provided dQ is negative or null. Received {value}')
        self._dQ = value

    @property
    def dE(self) -> float:
        return self._dE

    @dE.setter
    def dE(self, value: float):
        if not value > 0:
            raise AttributeError(f'Provided dE is negative or null. Received {value}')
        self._dE = value

    def fit(self, q_length, energy, intensity, normalization, monitor) -> 'PowderBackground':
        """Bin the points (1-D arrays: |Q|, energy, counts, normalization, monitor) without
        signal. |Q| bins start at 0, energy bins extend 1.1 bins beyond the data. Returns self"""
        q_length, energy = np.asarray(q_length), np.asarray(energy)
        self.q_bins = np.arange(0, np.nanmax(q_length) + self.dQ*1.1, self.dQ)
        self.energy_bins = np.arange(np.nanmin(energy) - self.dE*1.1,
                                     np.nanmax(energy) + self.dE*1.1, self.dE)
        self.counts, self.normalization, self.monitor, self.bin_count = _tools.histogramdd(
            np.array([q_length, energy]).T, bins=(self.q_bins, self.energy_bins),
            weights=[intensity, normalization, monitor], returnCounts=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Empty bins divide by zero
            self.intensity = np.divide(self.counts*self.bin_count, self.monitor*self.normalization)
        return self

    def sample(self, q_length, energy, normalization=None, monitor=None) -> FloatArray:
        """Background at points (1-D arrays of |Q| and energy). Without normalization and
        monitor: the normalised background intensity. With them: the background counts each
        point would have recorded, comparable to its measured counts. Points outside the binned
        range get 0 (NaN where the model's bin holds no points)"""
        if self.q_bins is None:
            raise AttributeError('No background found: fit the model first')
        q_index = _tools.findFlattenedIndex(np.asarray(q_length).reshape(-1, 1), [self.q_bins]) - 1
        energy_index = _tools.findFlattenedIndex(np.asarray(energy).reshape(-1, 1),
                                                 [self.energy_bins]) - 1
        # The old model only checked the upper end: a point below the first bin edge (index -1)
        # took its background from the last bin
        inside = ((q_index >= 0) & (q_index < len(self.q_bins) - 1)
                  & (energy_index >= 0) & (energy_index < len(self.energy_bins) - 1))
        background = np.zeros(inside.shape, dtype=float)
        background[inside] = self.intensity[q_index[inside], energy_index[inside]]
        if normalization is not None:
            background[inside] *= np.asarray(normalization).reshape(background.shape)[inside]
        if monitor is not None:
            background[inside] *= np.asarray(monitor).reshape(background.shape)[inside]
        return background
