import copy
import warnings
from pathlib import Path
from operator import attrgetter
from collections.abc import Callable, Iterable, Iterator
import numpy as np

from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.AxisView import AxisView
from MJOLNIR.Data.Coordinates import convert_to_qxqy, convert_to_hkl, align_to_lattice
from MJOLNIR.Data import Cuts, Plotting
from MJOLNIR.Data.CutResult import CutResult
from MJOLNIR.Data.BackgroundModel import PowderBackground, AMBERBackground
from MJOLNIR.Data.DataReader import DataReader
from MJOLNIR.Data.DataReaderFactory import DataReaderFactory
from MJOLNIR.Data.InstrumentName import InstrumentName

# Anything that can be added to a DataSet: a reader, or the path of a file to read
DataFileLike = DataReader | str | Path

class DataSet:
    """A container of DataReader objects for different scans, measured on the same sample and
    all read by the same type of data reader. Data members of different scans can be returned
    as an AxisView object, which treats them as a single array contiguous along one axis, for
    the purposes of indexing and slicing."""
    instrument: InstrumentName | None
    reader_type: type[DataReader] | None
    _binning: int | None # None: the default binning of the reader type
    _axis: int # The contiguous axis for AxisView
    _readers: list[DataReader]
    _mask_object: object | None # Mask object (Mask.py) the masks were made from, if any
    _mask_reproducible: bool # Whether _mask_object (or nothing) reproduces the masks

    def __init__(self, instrument: InstrumentName | None = None,
                 files: DataFileLike | Iterable[DataFileLike] | None = None):
        """Currently we only allow for a data set containing data from one instrument. Without
        an instrument, it is taken from the first data file added"""
        self._reader_factory = DataReaderFactory()
        self.instrument = instrument
        self.reader_type = None
        self._binning = None
        if instrument is not None:
            self.reader_type = self._reader_factory.get_reader_type(instrument)
            self._binning = self.reader_type.default_binning
        self._axis = 0
        self._readers = []
        self._mask_object = None
        self._mask_reproducible = True
        self._use_default_mask = True
        self.frozen = False # A difference of data sets: cannot be rebinned or extended
        self.background_model = None
        self._background_settings = None # Method and arguments that fitted the model, to refit
        self._calibration_updates = [] # (calibration files, overwrite), also for later files
        if files is not None:
            self.extend(files)

    def set_binning(self, binning: int):
        """Recalibrate and transform all data files with a new binning. A mask set from a mask
        object is evaluated again; masks set from arrays no longer fit and are removed"""
        self._check_not_frozen('rebin')
        self._binning = binning
        for reader in self._readers:
            reader.set_instrument_calibration(binning)
            reader.transform()
        if not self._mask_reproducible:
            warnings.warn('Masks set from arrays do not fit the new binning and were removed')
            self._mask_reproducible = True
        elif self._mask_object is not None:
            self.mask = self._mask_object
        if self.background_model is not None:
            self._refit_background()

    def update_calibration(self, calibration_files: Path | str | list[Path | str],
                           overwrite: bool = False) -> list[int]:
        """Use calibrations from .calib files (Geometry.Instrument.generateCalibration) for
        all data files, also those added later (see DataReader.update_calibration). If the
        current binning is updated, the files are transformed again. Nothing is saved: use
        CameaDataWriter per file. Returns the binnings updated"""
        self._check_not_frozen('recalibrate')
        if isinstance(calibration_files, (str, Path)):
            calibration_files = [calibration_files]
        calibration_files = [Path(path) for path in calibration_files]
        updated = set()
        for reader in self._readers:
            updated.update(reader.update_calibration(calibration_files, overwrite))
        self._calibration_updates.append((calibration_files, overwrite))
        if self._binning in updated and len(self) > 0:
            self.set_binning(self._binning)
        return sorted(updated)

    # Adding data files

    def append(self, item: DataFileLike):
        """Add a data reader, or read a file with the instrument of the DataSet. Data files are
        calibrated with the binning of the DataSet and transformed when added"""
        self._check_not_frozen('add data files to')
        reader = self._to_reader(item)
        if self.reader_type is not None and type(reader) is not self.reader_type:
            raise TypeError(f'Cannot add {type(reader).__name__} "{reader.file.name}" to a '
                            f'DataSet of {self.reader_type.__name__}')
        if len(self) > 0 and not reader.sample == self.sample:
            raise ValueError(f'Sample of "{reader.file.name}" differs from the sample of '
                             f'"{self._readers[0].file.name}"')
        if self.reader_type is None:
            self.reader_type = type(reader)
            self.instrument = self._reader_factory.get_instrument(self.reader_type)
        if self._binning is None:
            self._binning = self.reader_type.default_binning
        updated = set()
        for calibration_files, overwrite in self._calibration_updates:
            updated.update(reader.update_calibration(calibration_files, overwrite))
        if reader.calibration is None or reader.binning != self._binning \
                or self._binning in updated:
            reader.set_instrument_calibration(self._binning)
            reader.transform()
        reader.use_default_mask = self._use_default_mask
        if self._mask_object is not None and self._mask_reproducible:
            reader.mask = self._mask_arrays(self._mask_object, [reader])[0]
        if self.background_model is not None: # From the existing model
            self._apply_background(reader)
        self._readers.append(reader)

    def extend(self, items: DataFileLike | Iterable[DataFileLike]):
        if isinstance(items, (DataReader, str, Path)):
            items = [items]
        for item in items:
            self.append(item)

    def add_file(self, path: Path):
        self.append(path)

    def add_files(self, paths: Iterable[Path]):
        self.extend(paths)

    def _to_reader(self, item: DataFileLike) -> DataReader:
        if isinstance(item, DataReader):
            return item
        if isinstance(item, (str, Path)):
            if self.instrument is None:
                raise ValueError(
                    f'An instrument is needed to read "{item}": pass instrument to DataSet')
            return self._reader_factory.create(self.instrument, Path(item))
        raise TypeError(f'Cannot add {type(item).__name__} to DataSet: expected a DataReader '
                        'or a file path')

    # Container

    def number_of_files(self) -> int:
        return len(self._readers)

    def _check_not_frozen(self, action: str):
        if self.frozen:
            raise ValueError(f'Cannot {action} a DataSet holding derived data (a difference of '
                             'data sets). Change the original data sets and subtract again')

    def __sub__(self, other: 'DataSet') -> 'DataSet':
        """Difference of two data sets with matching data files (same reader type, binning,
        number of files and data shapes). Per point, the counts of this set minus those of
        other scaled to this set's monitor: I1 - I2 * M1/M2, with the error
        sqrt(e1**2 + (e2 * M1/M2)**2) in intensity_error (e = sqrt(I) for measured counts).

        The result holds shallow copies of this set's readers: each has its own intensity and
        masks, and shares everything else (counts, coordinates, calibration, sample) with this
        set. Points masked in either set, or without monitor in other, are masked. The result
        is frozen: it cannot be rebinned or extended (rebin the originals and subtract again)"""
        if not isinstance(other, DataSet):
            return NotImplemented
        if other.reader_type is not self.reader_type:
            raise ValueError(f'Cannot subtract a DataSet of {getattr(other.reader_type, "__name__", None)} '
                             f'from one of {getattr(self.reader_type, "__name__", None)}')
        if len(other) != len(self):
            raise ValueError(f'Cannot subtract a DataSet of {len(other)} data files from one of '
                             f'{len(self)}')
        if other._binning != self._binning:
            raise ValueError(f'Cannot subtract data at binning {other._binning} from data at '
                             f'binning {self._binning}')
        readers = []
        for mine, theirs in zip(self._readers, other._readers):
            if mine.intensity.shape != theirs.intensity.shape:
                raise ValueError(f'"{theirs.file.name}" ({theirs.intensity.shape}) does not match '
                                 f'"{mine.file.name}" ({mine.intensity.shape})')
            difference = copy.copy(mine) # Shares all arrays; intensity and masks replaced below
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = mine.monitor/theirs.monitor
                difference.intensity = mine.intensity - theirs.intensity*scale
                difference.intensity_error = np.sqrt(mine.intensity_variance()
                                                     + theirs.intensity_variance()*scale**2)
            difference.user_mask = mine.user_mask | theirs.mask | ~np.isfinite(difference.intensity)
            difference.frozen = True
            difference.background = None # The first set's background does not fit a difference
            readers.append(difference)
        result = DataSet(instrument=self.instrument)
        result.reader_type = self.reader_type
        result._binning = self._binning
        result._use_default_mask = self._use_default_mask
        result._readers = readers
        result._mask_reproducible = False # The masks are arrays now
        result.frozen = True
        return result

    def __len__(self) -> int:
        return len(self._readers)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            subset = DataSet(instrument=self.instrument)
            subset.reader_type = self.reader_type
            subset._binning = self._binning
            subset.extend(self._readers[index])
            subset._mask_object = self._mask_object
            subset._mask_reproducible = self._mask_reproducible
            subset._use_default_mask = self._use_default_mask
            subset.frozen = self.frozen
            subset.background_model = self.background_model
            subset._background_settings = self._background_settings
            return subset
        try:
            return self._readers[index]
        except IndexError:
            raise IndexError(
                f'Index {index} is out of bounds for DataSet with length {len(self)}') from None

    def __delitem__(self, index: int | slice):
        if isinstance(index, int) and not -len(self) <= index < len(self):
            raise IndexError(
                f'Index {index} is out of bounds for DataSet with length {len(self)}')
        del self._readers[index]

    def __iter__(self) -> Iterator[DataReader]:
        return iter(self._readers)

    def __reversed__(self) -> Iterator[DataReader]:
        return reversed(self._readers)

    def __repr__(self) -> str:
        names = ', '.join(reader.file.name for reader in self)
        return f'DataSet([{names}])'

    @property
    def sample(self):
        """The sample of the first data file, which defines the frame of the DataSet"""
        if len(self) == 0:
            return None
        return self._readers[0].sample

    @property
    def samples(self) -> list:
        """The sample of each data file: the same crystal, but each with its own orientation"""
        return [reader.sample for reader in self._readers]

    # Coordinates

    def convert_to_qxqy(self, hkl) -> np.ndarray:
        """Convert HKL point(s) to Qx, Qy in the frame of the DataSet (its first file)"""
        return convert_to_qxqy(self.sample, hkl)

    def convert_to_hkl(self, qxqy) -> np.ndarray:
        """Convert Qx, Qy point(s) in the frame of the DataSet (its first file) to HKL"""
        return convert_to_hkl(self.sample, qxqy)

    def rotations(self) -> list[np.ndarray]:
        """2x2 matrix per data file that rotates its Qx, Qy into the frame of the first file"""
        first = self.sample.RotMat
        return [np.dot(first.T, sample.RotMat) for sample in self.samples]

    def gather_q(self, align_to_lattice: bool = False, mask=None) -> tuple[AxisView, AxisView]:
        """Qx and Qy of the unmasked points (as gather), with each data file rotated into the
        frame of the first, so that the same HKL has the same Qx, Qy in every file.

        With align_to_lattice, each file is instead rotated by its own sample's RotMat, into
        the frame in which Qx, Qy follow the reciprocal lattice (as the old plotQPlane did;
        see Coordinates.align_to_lattice for the matching sample). mask as in gather"""
        qx, qy = self.gather(ArrayType.QX, mask=mask), self.gather(ArrayType.QY, mask=mask)
        rotations = ([sample.RotMat for sample in self.samples] if align_to_lattice
                     else self.rotations())
        rotated = [np.einsum('ij,j...->i...', rotation, np.array([x, y]))
                   for rotation, x, y in zip(rotations, qx.arrays, qy.arrays)]
        return (AxisView([r[0] for r in rotated], axis = 0),
                AxisView([r[1] for r in rotated], axis = 0))

    # Masking

    @property
    def mask(self) -> list[np.ndarray]:
        """Mask of each data file: True excludes a point. Can be set to a list of arrays (one
        per file), one array for every file, or a mask object from Mask.py. A mask object is
        kept, and evaluated again when the binning changes or files are added.

        What is set here is the user's mask: each file also excludes the points its
        instrument cannot use (its default mask), unless use_default_mask is False"""
        return [reader.mask for reader in self._readers]

    @property
    def use_default_mask(self) -> bool:
        """Whether the instruments' default masks (e.g. broken detectors) apply"""
        return self._use_default_mask

    @use_default_mask.setter
    def use_default_mask(self, use: bool):
        self._use_default_mask = use
        for reader in self._readers:
            reader.use_default_mask = use

    @mask.setter
    def mask(self, mask):
        for reader, array in zip(self._readers, self._mask_arrays(mask, self._readers)):
            reader.mask = array
        self._mask_object = mask if self._is_mask_object(mask) else None
        self._mask_reproducible = self._mask_object is not None

    def append_mask(self, mask):
        """Mask more points, keeping those already masked. Takes the same input as mask"""
        for reader, array in zip(self._readers, self._mask_arrays(mask, self._readers)):
            reader.user_mask = reader.user_mask | array
        if self._is_mask_object(mask) and self._mask_reproducible:
            self._mask_object = (mask if self._mask_object is None
                                 else self._mask_object + mask)
        else:
            self._mask_object = None
            self._mask_reproducible = False

    def remove_masking(self):
        """Remove the user's mask; the default masks stay (see use_default_mask)"""
        for reader in self._readers:
            reader.user_mask = np.zeros(reader.intensity.shape, dtype=bool)
        self._mask_object = None
        self._mask_reproducible = True

    @staticmethod
    def _is_mask_object(mask) -> bool:
        from MJOLNIR.Data.Mask import MaskingObject # Mask.py imports this module
        return isinstance(mask, MaskingObject)

    def _mask_arrays(self, mask, readers: list[DataReader]) -> list[np.ndarray]:
        """One boolean array per reader, checked against the shape of its data"""
        if self._is_mask_object(mask):
            subset = DataSet(instrument=self.instrument)
            subset.reader_type = self.reader_type
            subset._binning = self._binning
            subset._readers = list(readers)
            arrays = mask(subset)
        elif isinstance(mask, (list, tuple)) or (
                isinstance(mask, np.ndarray) and mask.dtype == object):
            if len(mask) != len(readers):
                raise ValueError(f'Got {len(mask)} masks for {len(readers)} data files')
            arrays = list(mask)
        else:
            array = np.asarray(mask, dtype=bool)
            shape = readers[0].intensity.shape if readers else ()
            if array.shape == (len(readers), *shape): # One array per file, stacked
                arrays = list(array)
            else: # The same array for every file
                arrays = [array] * len(readers)
        result = []
        for reader, array in zip(readers, arrays):
            array = np.asarray(array, dtype=bool)
            if array.shape != reader.intensity.shape:
                raise ValueError(f'Mask of shape {array.shape} does not fit "{reader.file.name}" '
                                 f'with data of shape {reader.intensity.shape}')
            result.append(array)
        if result and all(array.all() for array in result):
            warnings.warn('The mask excludes every point')
        return result

    # Contiguous arrays

    def get_view(self, arr_type: ArrayType | str) -> AxisView:
        """One array of all data files, contiguous along the axis of the DataSet"""
        arr_type = ArrayType(arr_type)
        if len(self) == 0:
            raise ValueError('DataSet contains no data files')
        get = attrgetter(arr_type.value)
        arrays = [get(reader) for reader in self._readers]
        if any(array is None for array in arrays):
            raise AttributeError(f'{arr_type.value} is not available in every data file')
        return AxisView([np.atleast_1d(array) for array in arrays], axis = self._axis)

    def gather(self, arr_type: ArrayType | str, mask=None) -> AxisView:
        """The unmasked points of one quantity of all data files: per file, the points where
        its mask is False, flattened. view.arrays gives them per file, np.asarray(view) joined,
        and view.offsets[1:] the boundaries between files.

        mask replaces the files' masks for this call only (in any form ds.mask takes: arrays,
        one array per file, or a mask object); the DataSet is not changed"""
        arr_type = ArrayType(arr_type)
        if len(self) == 0:
            raise ValueError('DataSet contains no data files')
        get = attrgetter(arr_type.value)
        masks = ([reader.mask for reader in self._readers] if mask is None
                 else self._mask_arrays(mask, self._readers))
        arrays = []
        for reader, reader_mask in zip(self._readers, masks):
            array = get(reader)
            if array is None:
                raise AttributeError(f'{arr_type.value} is not available in every data file')
            if np.shape(array) != reader_mask.shape:
                raise ValueError(f'{arr_type.value} does not have the shape of the data '
                                 f'{reader_mask.shape}, so it cannot be masked; use get_view')
            arrays.append(np.asarray(array)[~reader_mask])
        return AxisView(arrays, axis = 0)

    # Background

    def generate_background_model(self, background_mask=None, dQ: float = 0.025,
                                  dE: float = 0.05, combine_foreground_mask: bool = False):
        """Fit a powder-average background (BackgroundModel.PowderBackground) and give every
        data file its background counts per point (ArrayType.BACKGROUND), as the old
        generateBackgroundModel did.

        Kwargs:

            - background_mask: Points to leave out of the fit, i.e. the signal (a mask object,
              arrays, ...: as ds.mask takes). It replaces the DataSet's mask for the fit; None
              uses the DataSet's mask (default None)

            - dQ, dE (float): Bin sizes of the powder average (default 0.025 1/AA, 0.05 meV)

            - combine_foreground_mask (bool): Also leave out the points masked in the DataSet,
              e.g. spurions or broken detectors (default False)

        The model is fitted again when the binning changes, unless background_mask was given
        as arrays. Returns the model (also ds.background_model)"""
        self._check_not_frozen('fit a background to')
        masks = self._background_masks(background_mask, combine_foreground_mask)
        qx, qy = self.gather(ArrayType.QX, mask=masks), self.gather(ArrayType.QY, mask=masks)
        gather = lambda arr_type: np.asarray(self.gather(arr_type, mask=masks))
        self.background_model = PowderBackground(dQ, dE).fit(
            np.linalg.norm([np.asarray(qx), np.asarray(qy)], axis=0), gather(ArrayType.ENERGY),
            gather(ArrayType.INTENSITY), gather(ArrayType.NORMALIZATION),
            gather(ArrayType.MONITOR))
        self._background_settings = ('generate_background_model', dict(
            background_mask=background_mask, dQ=dQ, dE=dE,
            combine_foreground_mask=combine_foreground_mask))
        for reader in self._readers:
            self._apply_background(reader)
        return self.background_model

    def generate_amber_background(self, dQx: float = 0.03, dQy: float = 0.03, dE: float = 0.05,
                                  background_mask=None, combine_foreground_mask: bool = False,
                                  l: float | None = None, mu: float | None = None,
                                  beta: float = 1.0, max_radius: float | None = None,
                                  n_bins: int | None = None, n_epochs: int = 20,
                                  verbose: bool = True):
        """Fit a background with AMBER (BackgroundModel.AMBERBackground; needs the AMBER-ds4ms
        package) and give every data file its background counts per point
        (ArrayType.BACKGROUND), as the old AMBERBackground with applyAMBER did.

        The points are binned in voxels of about dQx * dQy * dE (Cuts.bin_3d, aligned to the
        reciprocal lattice, as the old model did); AMBER separates the signal from a background
        depending on |Q| and energy.

        Kwargs:

            - dQx, dQy, dE (float): Voxel size (default 0.03 1/AA, 0.03 1/AA, 0.05 meV)

            - background_mask, combine_foreground_mask: Points to leave out of the voxels, as
              for generate_background_model (default: the DataSet's mask). The old model
              accepted a mask, but did not use it

            - l, mu, beta, max_radius, n_bins: See AMBERBackground

            - n_epochs (int): Largest number of AMBER iterations (default 20)

            - verbose (bool): Print the loss of each iteration (default True)

        Refitted when the binning changes, as generate_background_model. Returns the model
        (also ds.background_model)"""
        self._check_not_frozen('fit a background to')
        masks = self._background_masks(background_mask, combine_foreground_mask)
        binned = Cuts.bin_3d(**self._points(align_to_lattice=True, mask=masks),
                             dx=dQx, dy=dQy, dz=dE)
        self.background_model = AMBERBackground(l=l, mu=mu, beta=beta, max_radius=max_radius,
                                                n_bins=n_bins).fit(binned, n_epochs=n_epochs,
                                                                   verbose=verbose)
        self._background_settings = ('generate_amber_background', dict(
            dQx=dQx, dQy=dQy, dE=dE, background_mask=background_mask,
            combine_foreground_mask=combine_foreground_mask, l=l, mu=mu, beta=beta,
            max_radius=max_radius, n_bins=n_bins, n_epochs=n_epochs, verbose=verbose))
        for reader in self._readers:
            self._apply_background(reader)
        return self.background_model

    def remove_background_model(self):
        self.background_model = None
        self._background_settings = None
        for reader in self._readers:
            reader.background = None

    def _background_masks(self, background_mask, combine_foreground_mask: bool) -> list:
        if background_mask is None:
            return self.mask
        masks = self._mask_arrays(background_mask, self._readers)
        if combine_foreground_mask:
            masks = [np.logical_or(new, old) for new, old in zip(masks, self.mask)]
        return masks

    def _apply_background(self, reader):
        """Background counts at every point of a reader, from the background model"""
        q_length = np.linalg.norm([reader.qx.flatten(), reader.qy.flatten()], axis=0)
        reader.background = self.background_model.sample(
            q_length, reader.energy.flatten(), normalization=reader.normalization.flatten(),
            monitor=reader.monitor.flatten()).reshape(reader.intensity.shape)

    def _refit_background(self):
        """After a change of binning: fit the model again, if its mask can be evaluated again"""
        method, settings = self._background_settings
        mask = settings['background_mask']
        if mask is None or self._is_mask_object(mask):
            getattr(self, method)(**settings)
        else:
            warnings.warn('The background model was fitted with a mask given as arrays, which do '
                          'not fit the new binning: it was removed')
            self.remove_background_model()

    # Cuts and plots: gather the points and call Cuts and Plotting

    def _points(self, rotate: bool = True, align_to_lattice: bool = False,
                background_subtraction: bool = False, mask=None, errors: bool = False) -> dict:
        """The unmasked points, as the arrays Cuts takes. Qx, Qy rotated into the first file's
        frame (rotate), each file's own lattice frame (align_to_lattice), or not at all. With
        background_subtraction, also the background counts of the points. With errors, also
        intensity_error if any file has one (sqrt(intensity) for the others). mask as in
        gather"""
        if background_subtraction and self.background_model is None:
            raise AttributeError('No background model: call generate_background_model first')
        if rotate:
            qx, qy = self.gather_q(align_to_lattice=align_to_lattice, mask=mask)
        else:
            qx, qy = self.gather(ArrayType.QX, mask=mask), self.gather(ArrayType.QY, mask=mask)
        gather = lambda arr_type: np.asarray(self.gather(arr_type, mask=mask))
        points = dict(qx=np.asarray(qx), qy=np.asarray(qy), energy=gather(ArrayType.ENERGY),
                      intensity=gather(ArrayType.INTENSITY),
                      normalization=gather(ArrayType.NORMALIZATION),
                      monitor=gather(ArrayType.MONITOR))
        if background_subtraction:
            points['background'] = gather(ArrayType.BACKGROUND)
        if errors and any(reader.intensity_error is not None for reader in self._readers):
            masks = ([reader.mask for reader in self._readers] if mask is None
                     else self._mask_arrays(mask, self._readers))
            points['intensity_error'] = np.concatenate(
                [np.sqrt(reader.intensity_variance())[~reader_mask]
                 for reader, reader_mask in zip(self._readers, masks)])
        return points

    def _cut_positions(self, q1, q2, rlu: bool):
        """q1, q2 in Qx, Qy: converted from H, K, L with rlu"""
        if rlu:
            return self.convert_to_qxqy(q1), self.convert_to_qxqy(q2)
        return np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)

    def cut_1d(self, q1, q2, width: float, min_pixel: float, energy_min: float,
               energy_max: float, rlu: bool = True, extend: bool = False,
               constant_bins: bool = False, background_subtraction: bool = False) -> CutResult:
        """1D cut from q1 to q2 (H, K, L with rlu, else Qx, Qy), summed over energy_min to
        energy_max. As in the old cut1D, files are rotated into the first file's frame only
        with rlu. With background_subtraction, the background model is subtracted. See
        Cuts.cut_1d"""
        q1, q2 = self._cut_positions(q1, q2, rlu)
        points = self._points(rotate=rlu, background_subtraction=background_subtraction)
        return Cuts.cut_1d(**points, q1=q1, q2=q2, width=width, min_pixel=min_pixel,
                           energy_min=energy_min, energy_max=energy_max, sample=self.sample,
                           extend=extend, constant_bins=constant_bins)

    def cut_constant_q(self, q, width: float = 0.04, min_pixel: float = 0.1,
                       energy_min: float = -np.inf, energy_max: float = np.inf,
                       rlu: bool = True, constant_bins: bool = False,
                       background_subtraction: bool = False) -> CutResult:
        """Energy scan at constant Q (the old cut1DE): the points within width/2 of q (H, K, L
        with rlu, else Qx, Qy), binned in energy. width is the diameter, as cut_1d's width is
        the full width (the old cut1DE used it as the radius: its default 0.02 is width
        0.04 here). With rlu, files are rotated into the first file's frame, as in cut_1d (the
        old cut1DE did not). Errors of derived data (intensity_error, e.g. a difference of
        data sets) are used. With background_subtraction, the background model is subtracted.
        See Cuts.cut_constant_q"""
        q = self.convert_to_qxqy(q) if rlu else np.asarray(q, dtype=float)
        points = self._points(rotate=rlu, background_subtraction=background_subtraction,
                              errors=True)
        return Cuts.cut_constant_q(**points, q=q, width=width, min_pixel=min_pixel,
                                   energy_min=energy_min, energy_max=energy_max,
                                   sample=self.sample, constant_bins=constant_bins)

    def cut_qe(self, q1, q2, width: float, min_pixel: float, energy_bins=None,
               energy_min=None, energy_max=None, energy_step=None, rlu: bool = True,
               extend: bool = False, background_subtraction: bool = False) -> CutResult:
        """Q-E cut from q1 to q2 (H, K, L with rlu, else Qx, Qy), in energy_bins or from
        energy_min to energy_max in steps of energy_step. As in the old cutQE, files are rotated
        into the first file's frame only with rlu. With background_subtraction, the background
        model is subtracted. See Cuts.cut_qe"""
        q1, q2 = self._cut_positions(q1, q2, rlu)
        points = self._points(rotate=rlu, background_subtraction=background_subtraction)
        return Cuts.cut_qe(**points, q1=q1, q2=q2, width=width, min_pixel=min_pixel,
                           energy_bins=energy_bins, energy_min=energy_min,
                           energy_max=energy_max, energy_step=energy_step,
                           sample=self.sample, extend=extend)

    def cut_qe_line(self, q_points, energy_bins, width=0.1, min_pixel=0.01, rlu: bool = True,
                    background_subtraction: bool = False) -> CutResult:
        """Q-E cuts along a path of Q points (H, K, L with rlu, else Qx, Qy): one cut_qe per
        segment (the old cutQELine). energy_bins, width and min_pixel are shared, or given per
        segment. As in cut_qe, files are rotated into the first file's frame only with rlu.
        See Cuts.cut_qe_line"""
        q_points = np.atleast_2d(np.asarray(q_points, dtype=float))
        expected = 3 if rlu else 2
        if q_points.shape[1:] != (expected,):
            raise ValueError(f'q_points must have shape (n, {expected}) with rlu={rlu}; got '
                             f'{q_points.shape}')
        if rlu:
            q_points = np.array([self.convert_to_qxqy(q) for q in q_points])
        points = self._points(rotate=rlu, background_subtraction=background_subtraction)
        return Cuts.cut_qe_line(**points, q_points=q_points, energy_bins=energy_bins,
                                width=width, min_pixel=min_pixel, sample=self.sample)

    def cut_q_plane(self, energy_min=None, energy_max=None, energy_bins=None,
                    x_bin_tolerance: float = 0.05, y_bin_tolerance: float = 0.05,
                    rlu: bool = True, background_subtraction: bool = False) -> CutResult:
        """Constant-energy planes binned in Qx, Qy. With rlu, the points are aligned to the
        reciprocal lattice (for RLU axes), else left in each file's frame, as the old
        plotQPlane did. With background_subtraction, the background model is subtracted. See
        Cuts.cut_q_plane"""
        sample = align_to_lattice(self.sample) if rlu else self.sample
        points = self._points(rotate=rlu, align_to_lattice=True,
                              background_subtraction=background_subtraction)
        return Cuts.cut_q_plane(**points,
                                energy_bins=energy_bins, energy_min=energy_min,
                                energy_max=energy_max, x_bin_tolerance=x_bin_tolerance,
                                y_bin_tolerance=y_bin_tolerance, sample=sample)

    def bin_3d(self, dx: float, dy: float, dz: float, rlu: bool = True,
               background_subtraction: bool = False) -> CutResult:
        """Bin the points in voxels of Qx, Qy and energy, about dx * dy * dz in size. With rlu,
        the points are aligned to the reciprocal lattice, else left in each file's frame, as
        the old binData3D did. With background_subtraction, the background model is
        subtracted. result.grid(column) gives a column as a 3D array. See Cuts.bin_3d"""
        sample = align_to_lattice(self.sample) if rlu else self.sample
        points = self._points(rotate=rlu, align_to_lattice=True,
                              background_subtraction=background_subtraction)
        return Cuts.bin_3d(**points, dx=dx, dy=dy, dz=dz, sample=sample)

    def plot_cut_1d(self, q1, q2, width: float, min_pixel: float, energy_min: float,
                    energy_max: float, rlu: bool = True, extend: bool = False,
                    constant_bins: bool = False, background_subtraction: bool = False,
                    **plot_options):
        """1D cut (cut_1d) plotted with error bars along the cut (Plotting.plot_cut_1d, which
        takes plot_options such as ax, counts, show_energy, plot_foreground, plot_background
        and errorbar options). Returns the axes; the cut is ax.result"""
        result = self.cut_1d(q1, q2, width, min_pixel, energy_min, energy_max, rlu=rlu,
                             extend=extend, constant_bins=constant_bins,
                             background_subtraction=background_subtraction)
        return Plotting.plot_cut_1d(result, sample=self.sample, rlu=rlu, **plot_options)

    def generate_ufit_dataset(self, result: CutResult, rlu: bool = True):
        """A ufit dataset of a 1D cut of this data set (from cut_1d or cut_constant_q), with
        the instrument, experiment, title and file names as metadata (the old
        generateUFitDataset). See Export.to_ufit"""
        from MJOLNIR.Data.Export import to_ufit
        metadata = [reader._metadata for reader in self._readers]
        meta = dict(instrument=self._readers[0].instrument,
                    experiment=', '.join(str(m.get('experimentalIdentifier', 'UNKNOWN'))
                                         for m in metadata),
                    title=metadata[0].get('title', ''),
                    datafilename=', '.join(reader.file.name for reader in self._readers))
        return to_ufit(result, rlu=rlu, meta=meta)

    def view_3d(self, dx: float, dy: float, dz: float, rlu: bool = True,
                background_subtraction: bool = False, **viewer_options):
        """Interactive slices through the data binned in voxels of dx * dy * dz (bin_3d), the
        old View3D. viewer_options go to SliceViewer.SliceViewer (axis, counts, log, vmin, vmax,
        cmap, adjustable, on_cut, cut_width, cut_min_pixel, spurion_peaks, ...). Cuts drawn in
        the viewer are made on this data set unless on_cut is given. Returns the viewer"""
        from MJOLNIR.Data.SliceViewer import SliceViewer
        result = self.bin_3d(dx, dy, dz, rlu=rlu, background_subtraction=background_subtraction)
        Ef = np.concatenate([np.ravel(reader.calibration.Ef) for reader in self._readers])
        Ef = Ef[np.isfinite(Ef) & ~np.isclose(Ef, 0.0)]
        viewer_options.setdefault('Ei', np.unique(np.concatenate(
            [np.ravel(reader.Ei) for reader in self._readers])))
        viewer_options.setdefault('ef_limits', (Ef.min(), Ef.max()) if Ef.size else None)
        return SliceViewer(result, sample=self.sample, rlu=rlu, dataset=self, **viewer_options)

    def plot_cut_qe(self, q1, q2, width: float = 0.1, min_pixel: float = 0.05,
                    energy_bins=None, energy_min=None, energy_max=None, energy_step=None,
                    rlu: bool = True, extend: bool = False, background_subtraction: bool = False,
                    **plot_options):
        """Q-E cut (cut_qe) plotted as a colour map (Plotting.plot_cut_qe, which takes
        plot_options such as ax, colorbar, vmin, vmax, log). Returns the axes; the cut is
        ax.result"""
        result = self.cut_qe(q1, q2, width, min_pixel, energy_bins=energy_bins,
                             energy_min=energy_min, energy_max=energy_max,
                             energy_step=energy_step, rlu=rlu, extend=extend,
                             background_subtraction=background_subtraction)
        return Plotting.plot_cut_qe(result, sample=self.sample, rlu=rlu, **plot_options)

    def plot_cut_qe_line(self, q_points, energy_bins, width=0.1, min_pixel=0.01,
                         rlu: bool = True, background_subtraction: bool = False,
                         **plot_options):
        """Q-E cuts along a path of Q points (cut_qe_line) plotted side by side
        (Plotting.plot_cut_qe_line, which takes plot_options such as ax, vmin, vmax, log,
        ticks, plot_separator, colorbar). Returns the axes; the cuts are ax.result"""
        result = self.cut_qe_line(q_points, energy_bins, width=width, min_pixel=min_pixel,
                                  rlu=rlu, background_subtraction=background_subtraction)
        return Plotting.plot_cut_qe_line(result, sample=self.sample, rlu=rlu, **plot_options)

    def plot_q_plane(self, energy_min=None, energy_max=None, energy_bins=None,
                     x_bin_tolerance: float = 0.05, y_bin_tolerance: float = 0.05,
                     rlu: bool = True, background_subtraction: bool = False, **plot_options):
        """Constant-energy planes (cut_q_plane) plotted on reciprocal lattice axes
        (Plotting.plot_q_plane, which takes plot_options such as ax, colorbar, vmin, vmax,
        log). Returns the axes; the cut is ax.result"""
        result = self.cut_q_plane(energy_min=energy_min, energy_max=energy_max,
                                  energy_bins=energy_bins, x_bin_tolerance=x_bin_tolerance,
                                  y_bin_tolerance=y_bin_tolerance, rlu=rlu,
                                  background_subtraction=background_subtraction)
        return Plotting.plot_q_plane(result, sample=self.sample, rlu=rlu, **plot_options)

    # Sorting

    @staticmethod
    def default_sort_key(reader: DataReader) -> tuple:
        """Sort by lowest Ei, then |2theta|, then A3 scan direction, then A3 start"""
        direction = -np.sign(reader.A3[1] - reader.A3[0]) if len(reader.A3) > 1 else 0.0
        return (np.round(reader.Ei[0], 1), np.abs(np.round(reader.twotheta[0], 1)),
                direction, np.round(reader.A3[0], 2))

    def arg_auto_sort(self, key: Callable[[DataReader], tuple] | None = None) -> np.ndarray:
        """Indices that sort the data files by key (default: default_sort_key)"""
        if key is None:
            key = self.default_sort_key
        return np.array(sorted(range(len(self)), key=lambda i: key(self._readers[i])),
                        dtype=int)

    def auto_sort(self, key: Callable[[DataReader], tuple] | None = None):
        """Sort the data files by key (default: default_sort_key)"""
        self.apply_sorting(self.arg_auto_sort(key))

    def arg_sort_to(self, other: 'DataSet',
                    key: Callable[[DataReader], tuple] | None = None) -> np.ndarray:
        """Indices that order these data files like the matching files of another data set"""
        return self.arg_auto_sort(key)[np.argsort(other.arg_auto_sort(key))]

    def apply_sorting(self, indices: Iterable[int]):
        indices = list(indices)
        if sorted(indices) != list(range(len(self))):
            raise ValueError(f'Sorting indices {indices} are not a permutation of the '
                             f'{len(self)} data files')
        self._readers = [self._readers[i] for i in indices]
