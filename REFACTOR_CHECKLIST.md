# Refactoring checklist: DataFile / DataSet → readers + new DataSet

Tracks the port of `MJOLNIR/Data/DataFile.py` and the original `DataSet`
(`MJOLNIR/Data/DataSet_old.py`) to the new architecture: the `DataReader` hierarchy,
`DataReaderFactory`, the new `DataSet` (`MJOLNIR/Data/DataSet.py`) and `AxisView`.
Items are checked off as they are completed.

## Milestone 1: the CAMEA test script (reached 2026-10-17)

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
- [x] Subtraction of two data sets (by measured monitor): `ds1 - ds2`, a frozen DataSet
      sharing all but intensity and masks with `ds1`; milestone plots pixel-identical
- [x] Expose each file's sample (`ds.samples`; every reader already has its own), needed to
      rotate files into a common frame
- [x] `convertToQxQy`, `convertToHKL`, and rotating files into the first file's frame
      (`convert_to_qxqy`, `convert_to_hkl`, `gather_q`; identical to the old code)
- [x] `cutQE` with its 1D cut algorithm: pure functions `Cuts.cut_qe`, `Cuts.cut_1d`,
      returning a `CutResult`
- [x] `ds.cut_qe` (and `ds.cut_1d`): gather the points and call `Cuts`; also
      `ds.cut_q_plane`, `ds.plot_cut_qe`, `ds.plot_q_plane` (identical to the old code;
      `ds.plot_cut_1d` waits for a 1D plot in `Plotting`)
- [x] `cutQE` output compared with the old `DataSet` on the same data (before porting the
      plots): all columns identical for the milestone cut
- [x] `plotCutQE` (Q-E axes, colour limits): `Plotting.plot_cut_qe`; the milestone plot is
      pixel-identical to the old code's
- [x] `plotQPlane` (Q-plane axes from `RLUAxes`, colour limits): `Cuts.cut_q_plane` and
      `Plotting.plot_q_plane`; the milestone plot is pixel-identical to the old code's
- [x] Script switched to `from MJOLNIR.Data import DataSet`; plots match the references:
      pixel-identical to the old script's (`make_plots_old.py`) once both use the same font.
      The old plots' font came from importing `ufit`, which changes matplotlib's font on
      import; the new code does not import it, so its plots use matplotlib's default font

## Milestone 2: remove the old DataFile.py and DataSet_old.py

Reached when `MJOLNIR/Data/DataFile.py`, `MJOLNIR/Data/DataSet_old.py` and
`MJOLNIR/Data/BackgroundModel_old.py` are deleted, nothing in the repository imports them, and
the test suite passes. Every old feature is either ported or dropped on purpose. The porting
work is the open items in the sections further down; tick both.

### Decide what to port
- [ ] Answer the scope questions of the refactoring log: which old methods must keep working
      and for how long (questions 3 and 6), whether broken or unused features go (29-35:
      combined `h/k/l`, `cutELine`, `plotA3A4`, `save`/`load`, `createEmptyDataFile`,
      `shallowRead`, `saveHDF`, converted `.nxs` files, metadata-only fields). McStas files
      are decided: dropped for now (see "Later")
- [ ] Mark every open item in this file as "port" or "drop", checking each against what
      MJOLNIR-GUI uses (Milestone 3), so nothing the GUI needs is dropped

### Keep the comparisons with the old code
- [ ] Save the old code's output as reference test data before deleting it: conversion arrays
      for each instrument, `cut1D`, `cutQE`, `plotQPlane` values, `binData3D`, the powder and
      AMBER backgrounds, the milestone plots. The parity checks then run without the old code
- [ ] Replace `test/test_DataSetSmoke.py` (smoke tests of `DataSet_old`) with tests of the new
      `DataSet` for every ported operation
- [ ] Port or delete the tests pytest does not collect (`test/DataFile.py`, `test/DataSet.py`,
      `test/Sample.py` and the other `test/*.py` not named `test_*.py`); they need camea2018
      files that are not in `Data/` (question 37)
- [ ] One place for test data, available in CI (question 36); remove the hard-coded paths
      (`/home/zamaan/Projects/bambus` in `test_CameaDataReader.py` and
      `test_DataReaderFactory.py`, `~/projects/CoRhO4`, `~/projects/marmot+flatcone`)

### Port the remaining features (items further down)
- [ ] Files: converted `.nxs` (NXsqom), `createEmptyDataFile`, `shallowRead`, error for files
      without data. McStas is not ported here (see "Later")
- [ ] File metadata: monitor mode and preset, monitor fixes, counting time, `absoluteTime`,
      scan parameters and command, `startTime`/`endTime`, `title`, `comment`, detector and
      analyser selection (`dasel`)
- [ ] Calibration and sample: `updateCalibration`, calibration plots, `updateSampleParameters`
- [ ] Saving: `saveNXsqom`, DataSet `save`/`load`. `saveHDF` is replaced by the writers below
- [x] `NexusLayout` and `CameaLayout` (`MJOLNIR/Data/NexusLayout.py`), not used by anything
      yet. Each field is a `Field(path, aliases, units, dtype)`: readers are to try `path`
      then the aliases, writers write `path` with its units and dtype; groups are
      `Group(path, nx_class)`. The class dictionaries hold templates (`{entry}`,
      `{instrument}`, `{binning}` for calibration), filled in by `field`, `path`, `group`,
      `all_fields`, `calibration_fields(binning)`; misspelt placeholders and keys defined
      twice fail when the class is defined. CAMEA units, dtypes and `NX_class` values taken
      from the 2023 and 2024 NICOS files; `test/test_NexusLayout.py` checks them against
      the files and checks the paths agree with `CameaDataReader`'s keys
- [x] `CameaDataReader` reads through `CameaLayout` (`layout = CameaLayout`); its
      `nexus_keys`, `nexus_meta_keys` and `environment_keys` are removed.
      `NexusDataReader` reads through the layout (`getValue` takes the first of a field's
      paths in the file, and also returns groups; `environment_paths`, `metadata_paths`,
      `can_read`). Calibration, possible
      binnings and the NICOS check in `scan_monitor` come from the layout. Reader output
      identical before and after on all 20 CAMEA files (counts, angles, monitors,
      environment, metadata, every calibration, sample, `can_read`). Changes: the
      temperature is read from `sample/temperature` first, then `Ts`, `Ts/value`, `se_ts`,
      `T` (the reader tried `temperature` last; same for `magnetic_field` and `B`), which
      only matters for a file that has both; `can_read` accepts old files with
      `detector/counts`
- [x] `FlatconeLayout`, `MarmotLayout`, sharing `IllLayout` (`entry0`, Ei, scan command and
      scanned variables, ILL user metadata), checked against `043555_pastis.nxs` and
      `054724.nxs`; their readers use them (Marmot's `identify` too). `NexusLayout` now
      holds only what all NeXus files share; the PSI parts (control, monitors, sample
      angles and UB, `scancommand`, PSI metadata) moved to `CameaLayout`. `Group.nx_class`
      may be None (Flatcone's `monitor` and `sample/param` have none). The key tables, the
      fallback and `__init_subclass__` are removed from `NexusDataReader`, and the readers'
      `entry` attributes with them. Reader output identical before and after on both ILL
      files and all 20 CAMEA files, and every reader's `can_read` unchanged on every file
- [ ] ILL files store `{instrument}/A4/offset_value` (-178.65 and -179.8 degrees in the two
      test files), not read: `A4Offset` is 0 for Flatcone and Marmot. The old `loadMarmot`
      read it but then set `_A4Off = 0.0`. Confirm with ILL that it is not a correction to
      apply
- [x] In `CameaLayout`: the lists of `environment_keys` are aliases; old CAMEA
      `detector/counts` is an alias of `detector/data`
- [x] In the layouts: no duplicate keys (`monitor1`, `monitor2` replace `hdfMonitor`,
      `monitor`, `monitor_1`); `instrument` and `sample` are groups. Converted-file keys
      (`intensity`, `energy`, `normalization`) left out until converted files are decided
- [x] Calibration sub-layout: `calibration_group` and `calibration` (keys as in
      `InstrumentCalibration`). The files store no units for it, so none are written
- [x] Reader tests pass on the layouts (only the `nexus_keys['instrument']` checks became
      `layout.group('instrument').path`)
- [ ] `DataWriter` ABC (beside `DataReader`); `NexusWriter` on a layout; `CameaWriter`.
      NICOS text formats (Bambus, MultiFLEXX) get no writer
- [ ] Update-a-copy mode, the default: copy the source file, open the copy `'r+'`, replace
      only changed datasets, so fields MJOLNIR does not read survive. Never writes over the
      source file
- [ ] Save a new calibration with the update-a-copy mode (what the GUI uses `saveHDF` for)
- [ ] Round-trip test per writer: write, read back with the same instrument's reader, compare
- [ ] Full-file mode, only once `createEmptyDataFile` or NXsqom output is chosen for porting
- [ ] Cuts: `cut1DE`, `cutQELine`, `cutELine`, `cutPowder`, `cutRaw1D`, raw `extractData`,
      measurement errors (`I_err`), `cutQE` smoothing and scaled energy, `plotCoverage`
- [ ] Plots and axes: `plotCut1D`, `plotCut1DE`, `plotCutQELine`, `plotCutELine`,
      `plotCutPowder`, `plotRaw1D`, `plotA3A4`, `View3D` (on `bin_3d`), `createQAxis` /
      `createQEAxes`, `generate1DAxis(E)`, polar binning and `enlargen` in `plotQPlane`,
      click handlers and `_interactiveSettings` modes
- [ ] Normalisation: `absoluteNormalize` / `undoAbsoluteNormalize`, `symmetrize`
- [ ] Physics: resolution matrices, `findEi`, `checkSecondOrderElastic`
- [ ] Export: `generateUFitDataset`
- [ ] DataFile module constants the GUI and scripts read: `supportedRawFormats`,
      `supportedConvertedFormats`, `possibleAttributes`, `detectorLimits`, `analyzerLimits`

### Move the rest of MJOLNIR off the old API
- [ ] `Viewer1D.py`, `Viewer3D.py`, `Viewer3DPyQtGraph.py`, `RLUAxes.py`, `DraggableShapes.py`,
      `_interactiveSettings.py`
- [ ] `Statistics/CutObject.py`
- [ ] `Geometry/Instrument.py` (reads a vanadium file with `DataFile`)
- [ ] Command line scripts (Convert, 3DView, History, CalibrationInspector) and
      `CommandLineScripts/_tools.py`, with an `--instrument` option; make `test_commandline`
      run (it needs the scripts installed)
- [ ] `_tools.PointerArray`: remove once nothing uses it
- [ ] `Mask.py`: plotting combinations with `indexMask` or `CurratAxeMask`

### Documentation and release
- [ ] Rewrite the docs that use the old API: the 10 notebooks in `docs/notebooks`, the pages in
      `docs/Data`, the command line script pages
- [ ] API documentation of the new modules
- [ ] Migration guide, old to new: `DataSet(dataFiles=...)` + `convertDataFile` → `DataSet(
      instrument, files=...)`; `extractData()` → `gather`; `cut1D` → `cut_1d`;
      `generateBackgroundModel` → `generate_background_model`; result columns unchanged
- [ ] Replace `.travis.yml` with CI testing Python 3.11-3.14 (also in "Project setup")
- [ ] New major version, as the API changes
- [ ] Delete `DataFile.py`, `DataSet_old.py`, `BackgroundModel_old.py`; full test suite passes

## Milestone 3: interface with MJOLNIR-GUI

MJOLNIR-GUI (`~/projects/MJOLNIR-GUI`, MJOLNIRGui 1.0.0, requires `MJOLNIR>=1.2.1`) subclasses
the old classes (`GuiDataSet(DataSet)`, `GuiDataFile(DataFile)` in
`src/main/python/MJOLNIR_Data.py`) and calls the old API from its views. Reached when the GUI
runs on the new `DataSet` and readers, and every view works.

### Design
- [ ] Decide the boundary: the GUI wraps the new `DataSet` and readers instead of subclassing
      them, or MJOLNIR offers a small GUI-facing layer
- [ ] Instrument detection for files opened in the GUI: a factory helper that tries each
      reader's `can_read` (the core keeps requiring the instrument)
- [ ] Progress reporting while files are read and transformed (the GUI advances a progress bar
      per file): a callback in `DataSet.extend` and `set_binning`
- [ ] What the Qt list models need from a DataSet: `insert` and `pop` at a position, a file
      index, display names
- [ ] Messages raised while reading (e.g. masked broken tubes) delivered through `warnings` or
      logging, which the GUI can show (it passed a `printFunction`)

### File information the GUI shows
- [ ] Reader metadata for the file info panel and subtraction check: sample name, title,
      projection vectors, Ei, A3, 2θ, magnetic field and temperature (`SampleEnvironment`),
      comment, binning, scan command, scan steps and parameters, counting time, start and end
      time, file location, instrument, possible binnings
- [ ] Electronic log book: the same metadata for many files, from a quick read (`shallowRead`)
- [ ] Raw 1D view: detector and analyser selection, `calcualteDataIndexFromDasel`, the
      binning-1 calibration, `plotRaw1D`

### GUI views on the new API
- [ ] Data set manager: open files, binning (`set_binning`), sorting (`auto_sort`), masks
      (`mask`, `remove_masking`), `updateCalibration`, saving the calibration with
      `CameaWriter` in place of `saveHDF`
- [ ] Mask manager: Mask objects on the new DataSet (`ds.mask`, `append_mask`); saving masks
- [ ] Subtraction manager: `ds1 - ds2` (a frozen DataSet) and its A3/2θ/Ei match check
- [ ] Cut 1D manager: `cut_1d`, `cut1DE`, 1D plots from a stored cut, ufit export
      (`Gui1DCutObject`)
- [ ] Q-E line manager: `plot_cut_qe`, `plotCutQELine`, `calculatePosition` on the axes
- [ ] Q plane manager: `plot_q_plane`; `DraggableShapes.extractCut` on the new axes; energy
      range from `gather`
- [ ] View 3D manager: `View3D` on `bin_3d`
- [ ] Normalisation manager: `absoluteNormalize`
- [ ] Prediction tool and calculators: check which sample and UB functions they use
- [ ] `generateScripts.py`: generated scripts use the new API
- [ ] Saved GUI states (`GuiStates.py`): store the new objects; decide whether old states load

### Packaging and testing
- [ ] MJOLNIRGui requires the new MJOLNIR version, Python 3.11 or newer and numpy 2.3 or newer
- [ ] GUI smoke test: open the sample data and run each view's main action (e.g. pytest-qt)
- [ ] Milestone check: the GUI loads the milestone files and shows the Q-E cut and Q plane of
      `make_plots.py`, with the same values

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
- [x] McStas simulated files: dropped for now (decided 2026-10-19), not ported from
      `DataFile`. To be added later as `McStasReader.py` (see "Later")
- [x] Copy a reader (old copy constructor `DataFile(other)`, used by subtraction): shallow
      copies in `__sub__`
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
- [ ] `saveHDF`: write a raw HDF file. Replaced by `NexusWriter` (Milestone 2, "Port the
      remaining features"); not ported

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
- [x] Rotate files into the first file's frame when samples differ slightly (`rlu`):
      `gather_q`

### Masking
- [x] `mask` property, `append_mask`, `remove_masking` (snake case, like the rest of the
      new `DataSet`)
- [x] `calculateCurratAxeMask` on the whole set: `ds.mask = Mask.CurratAxeMask(...)`
- [x] Apply `Mask.py` objects; a mask object is evaluated again after `set_binning` and
      for files added later

### Background
- [x] `backgroundModel`, `backgroundIntensities`, `generateBackgroundModel`:
      `ds.generate_background_model` fits `BackgroundModel.PowderBackground` from the
      gathered points (`gather(..., mask=...)`) and stores each file's background counts in
      `reader.background` (`ArrayType.BACKGROUND`); refitted after `set_binning` (mask
      objects), applied to appended files, dropped by a difference. Identical to the old code
      on the milestone data; `Plotting.plot_powder_background` plots the model
- [x] `backgroundSubtraction` in cuts: `background` in `Cuts.cut_1d`/`cut_qe`/`cut_q_plane`,
      `background_subtraction` in the `DataSet` cut and plot methods. One column scheme for
      all cuts (the old `cut1D` one)
- [x] `AMBERBackground`: `BackgroundModel.AMBERBackground` fitted on `Cuts.bin_3d` voxels,
      `ds.generate_amber_background` (refitted on rebinning like the powder model),
      `cross_validation`. Needs the optional AMBER-ds4ms package (`MJOLNIR[AMBER]`, which
      pulls in torch). Same voxel intensities as the old model, and the same AMBER result
      with the old conventions; the fixes are listed under "Cut bugs fixed in the port"

### Arithmetic and normalisation
- [x] Monitor-scaled subtraction (`__sub__`): by measured monitor per point, not preset
- [ ] `absoluteNormalize` / `undoAbsoluteNormalize`
- [ ] `symmetrize`

### Coordinates
- [x] `convertToQxQy`, `convertToHKL` (`convert_to_qxqy`, `convert_to_hkl`)

### Cuts
- [x] `cut1D` (`Cuts.cut_1d`; identical to the old code apart from the `Int_err` fix)
- [ ] `cut1DE`
- [x] `cutQE` (`Cuts.cut_qe`; identical to the old code)
- [ ] Cut options not ported yet: measurement errors (`I_err`),
      `cutQE` smoothing and scaled energy, `ufit` output, `plotCoverage`
- [ ] `cutQELine`
- [ ] `cutELine`
- [ ] `cutPowder`
- [x] `binData3D` (`Cuts.bin_3d`, `ds.bin_3d`, `CutResult.grid`; `calculateBins` and
      `calculateGrid3D` reduced to `Cuts.grid_edges`, one edge array per axis). Identical
      to the old code. Not ported: `noApproximation` (returned 2 arrays instead of 4) and the
      debug branch `backgroundSubtraction == 2`
- [ ] `cutRaw1D`
- [ ] `extractData` (raw selection by A4 / Ef)
- [x] Pandas output columns (`pdNaming`): `CutResult.COLUMNS`

### Plotting and interactive figures
- [ ] `plotCut1D` (cut done: `Cuts.cut_1d` / `ds.cut_1d`; no plot yet)
- [ ] `plotCut1DE`
- [x] `plotCutQE` (`Plotting.plot_cut_qe`)
- [ ] `plotCutQELine`
- [ ] `plotCutELine`
- [ ] `plotCutPowder`
- [x] `plotQPlane` (`Cuts.cut_q_plane`, `Plotting.plot_q_plane`; options not ported yet:
      polar binning, `enlargen`, 3D axes, `scaleFunction`)
- [ ] `plotRaw1D`
- [ ] `plotA3A4` (with `boundaryQ`, `convexHullPoints`)
- [ ] `View3D`
- [ ] `createQAxis` / `createQEAxes`, `generate1DAxis(E)`
- [ ] CSV export from figures, click and hover handlers (Q-E plots: CSV export and hover
      done; printing the data at a clicked point and the `_interactiveSettings` modes not yet)

### Instrument physics
- [ ] `calculateResolutionMatrix`, `calculateResolutionMatrixAndVectors`
- [ ] `findEi`
- [ ] `checkSecondOrderElastic`

### Export and saving
- [ ] `generateUFitDataset` (ufit)
- [ ] `save` / `load`

## Bugs in the old code to fix, not port
Recorded as expected failures in `test/test_DataSetSmoke.py`, except the `saveHDF` items.
- [ ] Combined `h/k/l` point at attributes `H/K/L`
- [ ] `cutELine` parameter misspelt `EMn`; `plotCutELine` always fails
- [ ] `plotA3A4` calls the missing `voronoiTessellation`
- [x] `removeMasking` fails unless a `Mask` object was applied (new `remove_masking` works)
- [ ] `save` cannot pickle the data set
- [ ] `saveHDF` crashes on every NICOS-written CAMEA file (2023 and 2024 data): sample
      `x`/`y` are absent, read as `array(None, dtype=object)`, and h5py cannot store them
- [ ] `saveHDF` opens its target with mode `'w'`, so the crash leaves a truncated file; the
      GUI calibration update writes over the original raw file (`d.saveHDF(d.fileLocation)`),
      destroying it (1.6 MB file left as a 0.56 MB stub without counts, checked on a copy)
- [ ] `saveHDF` rebuilds the file from what `DataFile` read, in the pre-NICOS layout: drops
      `detector/data`, `total_counts`, `monitor_2`, the temperature and field logs and
      `qh/qk/ql`; adds fields the source lacks; `CameaDataReader` cannot read the result

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

## Cut bugs fixed in the port
- [x] `cut1D` truncated `Int_err` to whole counts (column types one place out of line), and
      stored `BinCount` as float
- [x] `plotCutQE` ignored its `counts` option (the values were overwritten with `Int`)
- [x] `binData3D` with background multiplied the background counts by normalization and
      monitor a second time; now subtracted like the other cuts, with the same columns
- [x] `AMBERBackground` ignored its `backgroundMask`; gave AMBER the lower voxel edges
      instead of the centres (each voxel moved outwards in |Q|); used `n_bins` |Q| edges for
      AMBER's `n_bins` bins (spacing off, last bin unused) and the lower energy edges only
      (last energy bin unused); overwrote `l` and `mu` with the estimates, so a refit kept
      stale ones
- [x] Background model: points below the first |Q| or energy bin edge took their background
      from the last bin (index -1); they now get 0, like points above the range
- [x] `cutQE` with background: `intensity` held counts minus background, `Int_err` was the
      root of that difference (NaN where the background is larger), and `Int_Bg`/`Int_Fg`
      held counts, not normalised values. Now as `cut1D`: `intensity` the counts,
      `intensity_Bg`/`intensity_Fg` the counts, `Int_Bg`/`Int_Fg` normalised, `Int_err` from
      both counts. `plotQPlane` with background had no background columns at all

## Code still using the old API
- [ ] `Mask.py`: imports switched to `DataReader` / new `DataSet` (masks on `qx/qy`,
      `h/k/l`, `energy`, `A3` work); metaclass replaced by `ABC`; coordinates accept
      `ArrayType` or strings. `CurratAxeMask` works on readers; masks apply through
      `DataSet.mask`. Still to do: the bugs listed under "Mask.py bugs"
- [x] `BackgroundModel.py`: the powder model is the new `BackgroundModel.py`; the old file
      is kept as `BackgroundModel_old.py` for `DataSet_old`
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
- [x] numpy 2.3 or newer required (`setup.cfg`, `requirements.txt`, docs/Introduction.rst);
      the old `numpy<2.0` pin let Python 3.14 run with numpy 1.26, which does not support
      it and computes wrong results there (`b -= 0.5*a` inside a function also changed `a`
      for arrays over 256 kB; the old `calculateGrid3D` gave wrong edges because of it).
      Code changes: removed names `np.string_` → `np.bytes_` and `np.NaN` → `np.nan`;
      `boxMask` rotation angle made a scalar; `array.shape = ...` (deprecated in numpy 2.5)
      replaced by `reshape` throughout, and no-op `array.dtype = bool` removed. With numpy
      2, angles derived from the files' single-precision values round slightly differently
      (NEP 50 promotion; ~5e-7 degrees in the sample misalignment), so a point on a bin
      edge can change bin; the old code changes identically. Tested with numpy 2.5.4
      (Python 3.14) and 2.4.6 (Python 3.11)

## Later (after Milestone 3)
- [ ] McStas simulated files: `MJOLNIR/Data/McStasReader.py`, a `McStasReader` derived from
      `DataReader`, registered in `DataReaderFactory`. Written new rather than ported: the
      old `DataFile` code can serve as a reference for the file layout until it is deleted
      (Milestone 2), and from git history after that

## Finish
Removing `DataFile.py` and `DataSet_old.py` is Milestone 2.
