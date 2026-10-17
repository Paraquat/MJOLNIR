# Refactoring checklist: DataFile / DataSet → readers + new DataSet

Tracks the port of `MJOLNIR/Data/DataFile.py` and the original `DataSet`
(`MJOLNIR/Data/DataSet_old.py`) to the new architecture: the `DataReader` hierarchy,
`DataReaderFactory`, the new `DataSet` (`MJOLNIR/Data/DataSet.py`) and `AxisView`.
Items are checked off as they are completed.

## Milestone 1: the CAMEA test script

`~/projects/camea_test/make_plots.py` (the original author's `TestScript.ipynb` as a script)
loads CAMEA scans 2630-2641 and saves a Q-E cut and a Q plane, each also for the data set
minus a copy of itself. The milestone is reached when it runs with the new `DataSet` and its
plots match the reference images next to it. Several items repeat ones further down; tick both.

- [x] Script that makes the 4 plots with the old `DataSet` (`DataSet_old`); output matches the
      reference images
- [x] Load the 12 CAMEA files into a `DataSet` (reader, factory, calibration, transform)
- [x] Energy transfer, qx/qy, h/k/l, intensity, normalization per file
- [x] Default binning 8 for CAMEA, as the old conversion chose
- [x] Monitor and normalization spread to the shape of the data (`monitor`,
      `normalization`; identical to the old conversion for all 5 instruments)
- [x] CAMEA masks: first 2 pixels at binning 8, broken tubes (default masks, kept apart
      from the user's mask)
- [x] Combined data across files without masked points (the old `extractData()`), e.g. for
      the largest energy transfer: `ds.gather(ArrayType.X)`, identical to the old arrays and
      `maskIndices` on the 12 milestone files
- [x] `magneticField` (used in the plot title): `reader.environment.magnetic_field`,
      `ds.get_view(ArrayType.MAGNETIC_FIELD)`
- [ ] Subtraction of two data sets (by measured monitor)
- [ ] One sample per file in the `DataSet` (needed to rotate files into a common frame)
- [ ] `convertToQxQy`, `convertToHKL`, and rotating files into the first file's frame
- [ ] `cutQE` with its 1D cut algorithm
- [ ] `cutQE` output compared with the old `DataSet` on the same data (before porting the
      plots)
- [ ] `plotCutQE` (Q-E axes, colour limits)
- [ ] `plotQPlane` (Q-plane axes from `RLUAxes`, colour limits)
- [ ] Script switched to `from MJOLNIR.Data import DataSet`; plots match the references

## Readers and conversion

### Done
- [x] `DataReader` base class: calibration, `transform()` (intensity, normalization,
      energy, qx/qy, h/k/l), pixel rebinning, sample from two reflections
- [x] `NexusDataReader`, `CameaDataReader`
- [x] `NICOSDataReader`, `BAMBUSDataReader`, `MultiFLEXXDataReader` (MultiFLEXX only
      checked against a synthetic file)
- [x] `FlatconeDataReader`, `MarmotDataReader` (Marmot needs an external calibration table)
- [x] `DataReaderFactory` with instrument ↔ reader type lookup
- [x] Default binning per reader type (`default_binning`: 8 for CAMEA, 1 otherwise)

### File handling
- [x] Readers give a clear error when given the wrong kind of file
- [x] Optional `can_read(path)` per reader, for the GUI to detect the instrument (from the
      old `getInstrument`, `isMultiFLEXX`, `checkNICOS`). Not used by the core: pipelines
      and `DataSet` always state the instrument. MultiFLEXX only checked against a
      synthetic file
- [ ] Read converted `.nxs` (NXsqom) files
- [ ] McStas simulated files
- [ ] Copy a reader (old copy constructor `DataFile(other)`, used by subtraction)
- [ ] `createEmptyDataFile`: simulated file for planning/resolution
- [ ] `shallowRead`: quick read of a few parameters (history tool)
- [ ] Clear error for files without data

### CAMEA data not read yet
- [ ] `MonitorMode`, `MonitorPreset`
- [ ] Monitor fixes: 2018 wrong monitor, extra value in files of running scans
- [ ] `Time` (counting time per step), `absoluteTime`, `totalCounts`
- [x] `temperature`, `magneticField`, `electricField`: `SampleEnvironment` (measured and
      setpoints, units from the file), identical to the old DataFile for all CAMEA test files
- [ ] Temperature from the NICOS log when no temperature is stored
- [ ] Scan parameters (`getScanParameter`: parameters, values, units)
- [ ] `scanCommand`, `protonBeam`, `singleDetector1/8`, `s2tRaw`
- [ ] Analyser / monochromator settings

### Conversion output
- [x] Monitor spread to the shape of the data
- [x] Default mask (nothing masked; reset by `transform()`)
- [x] CAMEA binning 8: mask the first 2 pixels
- [x] CAMEA: mask broken tubes (NaN calibration A4), with a message (general rule for all
      instruments: any pixel with a NaN A4, Ef or amplitude)
- [x] Marmot: mask Ef < 1
- [x] Normalization spread over all scan steps
- [ ] Detector/analyser selection (`dasel`) and the index calculation from it
- [x] `DataSet` default binning = the reader type's default binning

### Calibration and sample
- [ ] `updateCalibration`: external normalisation tables, several binnings, `overwrite`
- [ ] Calibration plots: `plotA4`, `plotEf`, `plotEfOverview`, `plotNormalization`
- [ ] `updateSampleParameters`: new unit cell, recalculate UB
- [x] `calculateCurratAxeMask`: spurion masking, moved to `Mask.py` (works for every reader)

### Saving
- [ ] `saveNXsqom`: save converted data
- [ ] `saveHDF`: write a raw HDF file

### Utilities
- [ ] Compare files (`__eq__`, `difference`)
- [ ] `__str__`, `shape` / `size`
- [ ] Lists of supported instruments and formats

## DataSet

### Done
- [x] Container: list access, slicing, iteration, `append` / `extend` / `del`
- [x] One reader type and one sample per `DataSet`
- [x] Sorting: `auto_sort`, `arg_auto_sort`, `arg_sort_to`, `apply_sorting`
- [x] Files calibrated and transformed when added; `set_binning` re-transforms
- [x] Arrays across files as one contiguous array (`get_view` → `AxisView`)
- [x] `ArrayType` as a `StrEnum` (own module), used by `get_view` and as mask coordinates
- [x] `AxisView`: accept numpy integer indices
- [x] `AxisView`: iteration returns float64 like indexing
- [x] `AxisView`: clear `ValueError` when built from no arrays

### Files and setup
- [ ] Normalisation and calibration files as inputs; converted files as inputs
- [ ] `settings` dict and extra keyword arguments
- [ ] Save converted files (`convertDataFile` with `saveFile`, `saveLocation`,
      `deleteOnConvert`)
- [ ] `updateCalibration`, `updateSampleParameters` for the whole set

### Combined data
- [x] Drop masked points when extracting data (old `extractData()`), `maskIndices`
      (`gather`; boundaries are `view.offsets[1:]`)
- [ ] `I_err`
- [ ] Rotate files into the first file's frame when samples differ slightly (`rlu`)

### Masking
- [x] `mask` property, `append_mask`, `remove_masking` (snake case, like the rest of the
      new `DataSet`)
- [x] `calculateCurratAxeMask` on the whole set: `ds.mask = Mask.CurratAxeMask(...)`
- [x] Apply `Mask.py` objects; a mask object is evaluated again after `set_binning` and
      for files added later

### Background
- [ ] `backgroundModel`, `backgroundIntensities`, `generateBackgroundModel`
- [ ] `backgroundSubtraction` in cuts

### Arithmetic and normalisation
- [ ] Monitor-scaled subtraction (`__sub__`)
- [ ] `absoluteNormalize` / `undoAbsoluteNormalize`
- [ ] `symmetrize`

### Coordinates
- [ ] `convertToQxQy`, `convertToHKL`

### Cuts
- [ ] `cut1D`
- [ ] `cut1DE`
- [ ] `cutQE`
- [ ] `cutQELine`
- [ ] `cutELine`
- [ ] `cutPowder`
- [ ] `binData3D` (with `calculateGrid3D`, `calculateBins`)
- [ ] `cutRaw1D`
- [ ] `extractData` (raw selection by A4 / Ef)
- [ ] Pandas output columns (`pdNaming`)

### Plotting and interactive figures
- [ ] `plotCut1D`
- [ ] `plotCut1DE`
- [ ] `plotCutQE`
- [ ] `plotCutQELine`
- [ ] `plotCutELine`
- [ ] `plotCutPowder`
- [ ] `plotQPlane`
- [ ] `plotRaw1D`
- [ ] `plotA3A4` (with `boundaryQ`, `convexHullPoints`)
- [ ] `View3D`
- [ ] `createQAxis` / `createQEAxes`, `generate1DAxis(E)`
- [ ] CSV export from figures, click and hover handlers

### Instrument physics
- [ ] `calculateResolutionMatrix`, `calculateResolutionMatrixAndVectors`
- [ ] `findEi`
- [ ] `checkSecondOrderElastic`

### Export and saving
- [ ] `generateUFitDataset` (ufit)
- [ ] `save` / `load`

## Bugs in the old code to fix, not port
Recorded as expected failures in `test/test_DataSetSmoke.py`.
- [ ] Combined `h/k/l` point at attributes `H/K/L`
- [ ] `cutELine` parameter misspelt `EMn`; `plotCutELine` always fails
- [ ] `plotA3A4` calls the missing `voronoiTessellation`
- [x] `removeMasking` fails unless a `Mask` object was applied (new `remove_masking` works)
- [ ] `save` cannot pickle the data set

## Mask.py bugs
- [x] `parse()` fails on Python 3.13+ (writes to `locals()` before `eval`)
- [x] `__hash__` does not match `__eq__` (hashes a temporary `dict.values()` by identity)
- [x] `a - b` sets `b.negated` on the original mask
- [x] Combining masks (`+`, `*`) renames the input masks (now names copies)
- [x] `boxMask` without `corner4` cannot be cloned (NaN corners), so `-box` etc. fail
- [x] `RotationMatrix3D` without an axis crashes (`np.eye((3,3))`)
- [ ] `indexMask` / `CurratAxeMask` `plot` raise, so plotting combinations containing them fails
- [x] `test/Mask.py` is not collected by pytest, and defines `test_CurratAxeMask` twice
      (renamed to `test/test_Mask.py`; regression tests for the fixed bugs are in
      `test/test_MaskBehaviour.py`)
- [x] Clean-up: Python 2 style (`super(Class, self)`, `== True/False`, `not x in`),
      commented-out code, wrong `lineMask` docstring, `MultiMask` copies of inherited
      methods, trailing whitespace

## Code still using the old API
- [ ] `Mask.py`: imports switched to `DataReader` / new `DataSet` (masks on `qx/qy`,
      `h/k/l`, `energy`, `A3` work); metaclass replaced by `ABC`; coordinates accept
      `ArrayType` or strings. `CurratAxeMask` works on readers; masks apply through
      `DataSet.mask`. Still to do: the bugs listed under "Mask.py bugs"
- [ ] `BackgroundModel.py`
- [ ] `Viewer1D.py`, `Viewer3D.py`, `RLUAxes.py`, `DraggableShapes.py`
- [ ] `_interactiveSettings.py`
- [ ] `Statistics/CutObject.py`
- [ ] `_tools.py` (`PointerArray`)
- [ ] Command line scripts: Convert, 3DView, History, CalibrationInspector
- [ ] Command line scripts take an `--instrument` option instead of detecting it

## Project setup
- [x] Python 3.11 or newer required (`setup.cfg`, Read the Docs, docs/Introduction.rst);
      new code tested on Python 3.11, 3.12, 3.13 and 3.14. Needed `StrEnum`; the
      `np.NDArray` annotations (only working on 3.14) replaced by `numpy.typing.NDArray`
- [ ] `.travis.yml` still tests Python 3.5-3.9 on Travis CI: replace or remove
- [ ] `setup.cfg` pins numpy below 2: decide whether to allow numpy 2

## Finish
- [ ] Remove `DataFile.py` and `DataSet_old.py`
