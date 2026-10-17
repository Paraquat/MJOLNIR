# Refactoring checklist: DataFile / DataSet → readers + new DataSet

Tracks the port of `MJOLNIR/Data/DataFile.py` and the original `DataSet`
(`MJOLNIR/Data/DataSet_old.py`) to the new architecture: the `DataReader` hierarchy,
`DataReaderFactory`, the new `DataSet` (`MJOLNIR/Data/DataSet.py`) and `AxisView`.
Items are checked off as they are completed.

## Readers and conversion

### Done
- [x] `DataReader` base class: calibration, `transform()` (intensity, normalization,
      energy, Qx/Qy, h/k/l), pixel rebinning, sample from two reflections
- [x] `NexusDataReader`, `CameaDataReader`
- [x] `NICOSDataReader`, `BAMBUSDataReader`, `MultiFLEXXDataReader` (MultiFLEXX only
      checked against a synthetic file)
- [x] `FlatconeDataReader`, `MarmotDataReader` (Marmot needs an external calibration table)
- [x] `DataReaderFactory` with instrument ↔ reader type lookup
- [x] Default binning when calibrating a single reader (largest possible)

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
- [ ] `temperature`, `magneticField`, `electricField`
- [ ] Temperature from the NICOS log when no temperature is stored
- [ ] Scan parameters (`getScanParameter`: parameters, values, units)
- [ ] `scanCommand`, `protonBeam`, `singleDetector1/8`, `s2tRaw`
- [ ] Analyser / monochromator settings

### Conversion output
- [ ] Monitor spread to the shape of the data
- [ ] Default mask
- [ ] CAMEA binning 8: mask the first 2 pixels
- [ ] CAMEA: mask broken tubes (NaN calibration A4), with a message
- [ ] Marmot: mask Ef < 1
- [ ] Normalization spread over all scan steps (currently shape `(1, det, bins)`)
- [ ] Detector/analyser selection (`dasel`) and the index calculation from it
- [ ] Decide the `DataSet` default binning (now 1; the old default was the largest)

### Calibration and sample
- [ ] `updateCalibration`: external normalisation tables, several binnings, `overwrite`
- [ ] Calibration plots: `plotA4`, `plotEf`, `plotEfOverview`, `plotNormalization`
- [ ] `updateSampleParameters`: new unit cell, recalculate UB
- [ ] `calculateCurratAxeMask`: spurion masking

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
- [ ] Drop masked points when extracting data (old `extractData()`), `maskIndices`
- [ ] `I_err`
- [ ] Rotate files into the first file's frame when samples differ slightly (`rlu`)

### Masking
- [ ] `mask` property, `appendMask`, `removeMasking`
- [ ] `calculateCurratAxeMask`
- [ ] Apply `Mask.py` objects

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
- [ ] `removeMasking` fails unless a `Mask` object was applied
- [ ] `save` cannot pickle the data set

## Code still using the old API
- [ ] `Mask.py`
- [ ] `BackgroundModel.py`
- [ ] `Viewer1D.py`, `Viewer3D.py`, `RLUAxes.py`, `DraggableShapes.py`
- [ ] `_interactiveSettings.py`
- [ ] `Statistics/CutObject.py`
- [ ] `_tools.py` (`PointerArray`)
- [ ] Command line scripts: Convert, 3DView, History, CalibrationInspector
- [ ] Command line scripts take an `--instrument` option instead of detecting it

## Finish
- [ ] Remove `DataFile.py` and `DataSet_old.py`
