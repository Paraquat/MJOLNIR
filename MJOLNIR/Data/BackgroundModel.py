"""Background models, fitted from arrays of data points. Like Cuts, they do not know about
DataSet: DataSet.generate_background_model (or generate_amber_background) gathers the points,
fits a model, and stores the background of every data point in its readers.

Both models describe the background as a function of |Q| and energy, binned in q_bins and
energy_bins (RadialBackground): PowderBackground averages the points without signal,
AMBERBackground separates signal and background in voxels with the AMBER package.

The old model, written for the original DataSet, is in BackgroundModel_old.py.
"""
import numpy as np
import numpy.typing as npt
import warnings

from MJOLNIR import _tools

FloatArray = npt.NDArray[np.float64]

class RadialBackground:
    """Background as a function of |Q| and energy: intensity (normalised, like Int of a cut)
    in bins with edges q_bins and energy_bins, set when a model is fitted"""

    q_bins: FloatArray | None = None
    energy_bins: FloatArray | None = None
    intensity: FloatArray | None = None

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

class PowderBackground(RadialBackground):
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

class AMBERBackground(RadialBackground):
    """Background separated from the signal by AMBER (the optional AMBER-ds4ms package:
    pip install MJOLNIR[AMBER]).

    AMBER takes the normalised intensity in voxels of Qx, Qy and energy (Cuts.bin_3d), and
    splits it into a sparse signal and a background that depends on |Q| and energy only,
    smooth along |Q|: the background is radially binned from 0 to max_radius in n_bins bins,
    with the energy bins of the voxels. No mask is needed, though masked points are left out
    of the voxels.
    """

    def __init__(self, l: float | None = None, mu: float | None = None, beta: float = 1.0,
                 max_radius: float | None = None, n_bins: int | None = None):
        """Kwargs:

            - l (float): Weight of the signal's sparsity. None estimates it from the median
              absolute deviation of the intensity (default None)

            - mu (float): Weight of the signal's smoothness along energy. None estimates it
              from the variance of the intensity along energy (default None)

            - beta (float): Weight of the background's smoothness along |Q|; can be chosen
              with cross_validation (default 1.0)

            - max_radius (float): Largest |Q| of the radial bins [1/AA]. None covers all voxels
              with data (default None)

            - n_bins (int): Number of radial bins. None makes them about as wide as the
              voxels (default None)
        """
        self.l = l
        self.mu = mu
        self.beta = beta
        self.max_radius = max_radius
        self.n_bins = n_bins
        self.amber = None # The AMBER object of the last fit, with the signal in amber.X
        self.fitted_parameters = None # l, mu, beta, max_radius and n_bins of the last fit

    def _setup(self, binned):
        """An AMBER object holding the voxels of binned (a Cuts.bin_3d result), and the
        parameters to run it with"""
        try:
            from AMBER import background
        except (ModuleNotFoundError, ImportError) as error:
            raise ModuleNotFoundError('The AMBER package was not found. It can be installed '
                                      'with: pip install MJOLNIR[AMBER]') from error
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # Empty voxels divide by zero
            # As the old model: counts * points / (monitor * normalization), not background
            # subtracted
            intensity = np.divide(binned.grid('intensity')*binned.grid('BinCount'),
                                  binned.grid('Monitor')*binned.grid('Normalization'))
        qx_edges, qy_edges = binned.bins['qx'], binned.bins['qy']
        energy_edges = binned.bins['energy']
        centres = lambda edges: 0.5*(edges[:-1] + edges[1:])

        max_radius = self.max_radius
        if max_radius is None: # The outer corner of every voxel with data
            x = np.maximum(np.abs(qx_edges[:-1]), np.abs(qx_edges[1:]))
            y = np.maximum(np.abs(qy_edges[:-1]), np.abs(qy_edges[1:]))
            corners = np.linalg.norm(np.meshgrid(x, y, indexing='ij'), axis=0)
            max_radius = np.max(corners[np.any(np.isfinite(intensity), axis=2)])
        n_bins = self.n_bins
        if n_bins is None:
            n_bins = int(np.ceil(max_radius/max(np.diff(qx_edges)[0], np.diff(qy_edges)[0])))

        amber = background.background(dtype=intensity.dtype)
        amber.set_gridcell_size(dqx=np.diff(qx_edges)[0], dqy=np.diff(qy_edges)[0],
                                dE=np.diff(energy_edges)[0])
        # Voxel centres: the old model gave the lower edges, which moved each voxel outwards
        # in |Q| by up to half a voxel
        amber.set_binned_data(Qx=centres(qx_edges), Qy=centres(qy_edges),
                              E=centres(energy_edges), Int=intensity)
        amber.set_radial_bins(max_radius=max_radius, n_bins=n_bins)
        amber.set_variables()
        parameters = dict(l=amber.MAD_lambda() if self.l is None else self.l,
                          mu=amber.mu_estimator() if self.mu is None else self.mu,
                          beta=self.beta, max_radius=max_radius, n_bins=n_bins)
        return amber, parameters

    def fit(self, binned, n_epochs: int = 20, verbose: bool = True) -> 'AMBERBackground':
        """Run AMBER on the voxels of binned (a Cuts.bin_3d result), for at most n_epochs
        iterations (verbose: print the loss of each). Returns self"""
        self.amber, self.fitted_parameters = self._setup(binned)
        parameters = self.fitted_parameters
        self.amber.denoising(self.amber.Ygrid, lambda_=parameters['l'],
                             beta_=parameters['beta'], mu_=parameters['mu'],
                             n_epochs=n_epochs, verbose=verbose)
        # The radial bins of AMBER (the old model used n_bins edges instead of n_bins + 1, and
        # the lower energy edges only, so its last |Q| and energy bins were never used)
        self.q_bins = np.asarray(self.amber.r_range, dtype=float)
        self.energy_bins = np.asarray(binned.bins['energy'], dtype=float)
        self.intensity = np.asarray(self.amber.b, dtype=float).T # |Q| bins x energy bins
        return self

    def cross_validation(self, binned, beta, q: float = 0.75, n_epochs: int = 15,
                         verbose: bool = True) -> FloatArray:
        """Error of the background for each value of beta, on the voxels of binned whose
        intensity is above the quantile q (the lowest error is the best beta). Does not change
        the fitted model"""
        amber, parameters = self._setup(binned)
        return amber.cross_validation(q=q, beta_range=np.atleast_1d(np.asarray(beta, dtype=float)),
                                      lambda_=parameters['l'], mu_=parameters['mu'],
                                      n_epochs=n_epochs, verbose=verbose)
