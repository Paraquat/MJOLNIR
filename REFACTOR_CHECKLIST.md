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
- [x] Answer the scope questions of the refactoring log (questions 3 and 6, 12 Nov 2026).
      3: no aliases for the old method names; MJOLNIR-GUI is changed to call the new API in
      Milestone 3. 6: `DataFile.py` and `DataSet_old.py` are deleted at the end of this
      milestone, with no deprecation release
- [x] Broken or unused features (questions 29-35), decided 2026-10-19. Dropped: combined
      `h/k/l`, `cutELine`/`plotCutELine`, `plotA3A4`, DataSet `save`/`load`, converted
      `.nxs` files (reading, `saveNXsqom`, saving converted files), `createEmptyDataFile`,
      the metadata-only fields. Ported: `shallowRead`. Replaced: `saveHDF` (by
      `CameaDataWriter`). McStas: later (see "Later"). Reasons in the refactoring log
- [ ] Mark every open item in this file as "port" or "drop", checking each against what
      MJOLNIR-GUI uses (Milestone 3), so nothing the GUI needs is dropped

### Keep the comparisons with the old code
- [x] Save the old code's output as reference test data before deleting it (12 Nov 2026):
      `test/reference/*.npz` (3.2 MB), written by `test/reference/generate.py` (needs the old
      code), compared by `test/test_Reference.py` through `test/reference_data.py`.
      Conversion arrays of CAMEA 2024 (binning 8; binning 3 with an A3 offset), CAMEA 2023,
      Bambus, MultiFLEXX (synthetic file), Flatcone, Marmot (placeholder table); `cut1D`
      (three settings), `cut1DE`, `cutQE`, `cutQELine`, `cutPowder`, `cutRaw1D`,
      `plotQPlane` (data and mesh), `binData3D` (rlu on and off, two step sizes), the powder
      background (with and without the foreground mask) and AMBER (old conventions; run when
      AMBER and torch are installed), `updateSampleParameters`, `createQAxis` /
      `createQEAxes`, the physics (`findEi`, `checkSecondOrderElastic`, the resolution
      matrices and ellipses at binning 8; `physics.npz`, added 12 Nov 2026). The milestone plots: their Q-E cut and Q plane, of the data and of a
      difference. Large arrays are kept as summaries (shape, NaN count, sum, sum of squares,
      min, max, 5000 sampled values); cut results whole. Known differences are left out of
      the comparison where they are: Flatcone counts (the old conversion truncated them),
      Marmot's single-precision unit cell (UB ~1e-7) and its half-step scan offset, `cut1D`'s
      truncated `Int_err`, `Int_err` of a difference, `cutRaw1D` with one selection (below).
      The four tests that imported the old code (`test_CutPowder`, `test_CutConstantQ`,
      `test_PlotReadOut`, `test_UpdateSampleParameters`) moved to `test_Reference.py`. With
      the old modules blocked from importing, the whole suite but `test_DataSetSmoke.py`
      passes
- [ ] Replace `test/test_DataSetSmoke.py` (smoke tests of `DataSet_old`) with tests of the new
      `DataSet` for every ported operation
- [ ] Port or delete the tests pytest does not collect (`test/DataFile.py`, `test/DataSet.py`,
      `test/Sample.py` and the other `test/*.py` not named `test_*.py`); they need camea2018
      files that are not in `Data/` (question 37)
- [ ] One place for test data, available in CI (question 36); remove the hard-coded paths
      (`/home/zamaan/Projects/bambus` in `test_CameaDataReader.py` and
      `test_DataReaderFactory.py`, `~/projects/CoRhO4`, `~/projects/marmot+flatcone`)

### Port the remaining features (items further down)
- [x] Files. Done: `shallowRead`, error for files without data. Dropped: converted `.nxs`,
      `createEmptyDataFile`. McStas is not ported here (see "Later")
- [ ] File metadata: monitor preset, monitor fixes, `absoluteTime`, temperature from the
      NICOS log (questions 9, 11, 12, 41, 42 of the refactoring log). Done: CAMEA scan
      parameters, detector and analyser selection (`dasel`), monitor mode, `title`,
      `comment`, `startTime`/`endTime`, counting time (`reader.time`), scan command
- [x] Calibration and sample: `updateCalibration`, calibration plots,
      `updateSampleParameters`
- [x] Saving: `saveHDF` is replaced by the writers below. Dropped (2026-10-19):
      `saveNXsqom` (with converted files) and DataSet `save`/`load`
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
- [x] `DataWriter` ABC (`DataWriter.py`: `save(path, overwrite=False)`, refuses the
      reader's own file and, without `overwrite`, an existing file); `NexusDataWriter`
      (`NexusDataWriter.py`) on the reader's layout; `CameaDataWriter`
      (`CameaDataWriter.py`). Named like the readers. NICOS text formats (Bambus,
      MultiFLEXX) get no writer
- [x] Update-a-copy mode: copy the source file to a temporary file beside the target, open
      it `'r+'`, write the changes, move it into place (a failed save leaves no file).
      `NexusDataWriter.set(key, value)` writes any layout field. A dataset of the same shape
      and dtype is overwritten in place (keeps attributes and NICOS's `entry/data/*` hard
      links); otherwise replaced, keeping its attributes with their stored HDF5 types.
      Missing groups get their `NX_class` from the layout; `units` written only when
      missing or different
- [x] Save a new calibration: `CameaDataWriter(reader, {binning: calibration}).save(path)`
      (default: the reader's current calibration). Shapes checked against detectors x
      analysers x binning; binnings the file lacks are added (`calibN`, `NXcollection`)
- [x] Round-trip tests (`test/test_CameaDataWriter.py`): a changed calibration reads back, all
      other calibrations and data unchanged, no path added or dropped, hard link kept; a new
      binning; wrong shapes refused with no file left; `NexusDataWriter` on a synthetic layout
      (in place, replaced, created, refused targets, failed save). Rewriting every
      calibration of all 20 CAMEA files leaves every dataset and attribute, with its type,
      unchanged
- [x] Full-file mode: Dropped (2026-10-19). Only `createEmptyDataFile` and NXsqom output
      needed it, and both are dropped
- [x] Cuts: done: `cut1DE` (`cut_constant_q`), `cutQELine` (`cut_qe_line`),
      `cutRaw1D` (`cut_raw_1d`), `cutPowder` (`cut_powder`), measurement errors (`I_err`)
      in every cut. Dropped: raw `extractData`; (2026-11-12) `cutQE` smoothing and scaled
      energy, `plotCoverage`
- [x] Plots and axes. Moved to Milestone 3 (2026-11-12): the `_interactiveSettings` modes
      (cutting from a plot). Dropped (2026-11-12): 3D axes in `plotQPlane`. Done: `plotCut1D` (`plot_cut_1d`), `plotCutQELine` (`plot_cut_qe_line`),
      `View3D` (`view_3d`, `SliceViewer`), `plotRaw1D` (`plot_raw_1d`), `plotCut1DE`
      (`plot_cut_constant_q`), `plotCutPowder` (`plot_cut_powder`), `createQAxis` /
      `createQEAxes` (`create_q_axis`, `create_qe_axes`), `generate1DAxis(E)` (inside the
      plots), the click read-out (`on_click`). Dropped (2026-11-12): polar binning,
      `enlargen` and `scaleFunction` in `plotQPlane`
- [x] Normalisation: done: `absoluteNormalize` / `undoAbsoluteNormalize`
      (`absolute_normalize`, `undo_absolute_normalize`). Dropped (2026-11-11): `symmetrize`
- [x] Physics (12 Nov 2026): resolution matrices, `findEi`, `checkSecondOrderElastic` (see
      "Instrument physics"). Resolution for the other instruments: see "Later"
- [x] Export: `generateUFitDataset` (`ds.generate_ufit_dataset`, `Export.to_ufit`)
- [x] DataFile module constants the GUI and scripts read. Moved to Milestone 3 (2026-11-12):
      `detectorLimits`, `analyzerLimits`. Done: `supportedRawFormats` is `DataReaderFactory().file_extensions()`
      (each reader's `file_extensions`; `CommandLineScripts/_tools.py` builds its file dialog
      filter from it and no longer imports `DataFile`). `possibleAttributes` is now
      `shallow_keys()` per reader; `supportedConvertedFormats` is dropped with converted files

### Move the rest of MJOLNIR off the old API
- [ ] `RLUAxes.py`. Moved to Milestone 3 (2026-11-12): `Viewer1D.py`, `DraggableShapes.py`,
      `_interactiveSettings.py` (none imports the old code)
- [ ] 3D viewer split (decided 2026-10-19; refactoring log, Design decisions): delete
      `Viewer3D.py` with the old code (replaced by `SliceViewer.py`); move
      `Viewer3DPyQtGraph.py` to MJOLNIR-GUI or drop it (it needs Qt and fails with pyqtgraph
      0.13: `QtGui.QWidget` and `QtGui.QMainWindow` moved to `QtWidgets`)
- [ ] `Statistics/CutObject.py`
- [x] `Geometry/Instrument.py`: no longer imports `DataFile`. `generateCalibration` built a
      `DataFile` of the vanadium file that nothing used (only a commented-out line); removed
- [ ] `Geometry/Instrument.py`, port or drop (nothing in the package imports it at run time):
      `generateCalibration` (writes the `.calib` files the readers load; keep, add a test
      pytest collects), `prediction`, `converterToA3A4`, `converterToQxQy`,
      `predictionInstrumentSupport` (keep: MJOLNIR-GUI's prediction tool),
      `calculateResoultionMatrix` (ported to `Data/Resolution.py`; drop with `DataSet_old`,
      its only caller besides the interactive modes),
      `convertToHDF` (unused: drop?), the `Instrument` class and XML geometry
      (`InstrumentXML.py`; only the uncollected `test/Instrument.py`, `test/InstrumentXML.py`:
      drop?), `getInstrumentInfo`, `timeEstimate`, `simpleDataSet`
- [x] Command line scripts (3DView, History, CalibrationInspector) and
      `CommandLineScripts/_tools.py` on the new API, each with `-i`/`--instrument` (default
      CAMEA). 3DView: `ds.view_3d` (`-b` defaults to the instrument's binning, not 8 for all);
      History: `shallow_read`; CalibrationInspector: the `Plotting.plot_calibration_*`
      functions. The scripts no longer run at import: `main(argv=None)` parses the arguments.
      `test/test_commandline.py` calls them in-process (no install needed; the remembered
      files go to a temporary settings file); the stale copy in `CommandLineScripts/` is
      deleted. Convert: Dropped (2026-11-10) with converted files, with its entry point and
      tutorial page (reasons in the refactoring log). History prints the whole `scanCommand`,
      which in NICOS-era CAMEA files is the batch script, as the old script did
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
- [ ] Change the GUI code to call the new API (`cut_1d`, `plot_cut_qe`, `view_3d`, ...,
      snake_case arguments, results as `CutResult` or axes): MJOLNIR keeps no aliases for the
      old method names (question 3, 12 Nov 2026)
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

### Moved from Milestone 2 (2026-11-12)
- [ ] `detectorLimits` / `analyzerLimits` (old `DataFile` module constants, read by the Raw 1D
      view): drop; the GUI uses `reader.n_detectors` / `reader.n_analysers` (`raw_counts`
      checks the range)
- [ ] Interactive cutting from plots (the `_interactiveSettings` modes: drawing rectangles and
      circles on Q-plane, Q-E and 3D plots to make cuts) in the GUI; the library plots get an
      `on_cut(kind, parameters)` hook on `plot_q_plane` and `plot_cut_qe`, as `SliceViewer`
- [ ] `Viewer1D.py`, `DraggableShapes.py`, `_interactiveSettings.py`: on the new API, or moved
      to MJOLNIR-GUI. `_interactiveSettings.py` imports PyQt5 at import only to choose mouse
      cursors: use matplotlib's `canvas.set_cursor` (any backend), or move it to the GUI, so
      MJOLNIR does not need Qt (PyQt5 is not in `install_requires`). None of the three imports
      the old code, so deleting it in Milestone 2 does not break them

### File information the GUI shows
- [ ] Reader metadata for the file info panel and subtraction check: sample name, title,
      projection vectors, Ei, A3, 2θ, magnetic field and temperature (`SampleEnvironment`),
      comment, binning, scan command, scan steps and parameters, counting time, start and end
      time, file location, instrument, possible binnings
- [ ] Electronic log book: the same metadata for many files, from a quick read: on
      `DataReaderFactory().shallow_read(instrument, files, keys)`, which returns one dict
      per file (the old one returned lists); parameter list from `shallow_keys()`
- [ ] Raw 1D view: `reader.detector_selection` / `analyser_selection` (settable; the GUI
      keeps the originals itself), `ds.plot_raw_1d()`; `reader.raw_counts(detector,
      analyser)` replaces `calcualteDataIndexFromDasel` and the binning-1 calibration

### GUI views on the new API
- [ ] Data set manager: open files, binning (`set_binning`), sorting (`auto_sort`), masks
      (`mask`, `remove_masking`), `updateCalibration`, saving the calibration with
      `CameaDataWriter` in place of `saveHDF`. The GUI wrote into the raw file itself;
      `CameaDataWriter` saves to a new file, so the GUI must choose a name (e.g. a suffix) and
      open the new file. The "Converted" filter of the add-files dialog goes (converted files
      are dropped)
- [ ] Mask manager: Mask objects on the new DataSet (`ds.mask`, `append_mask`); saving masks
- [ ] Subtraction manager: `ds1 - ds2` (a frozen DataSet) and its A3/2θ/Ei match check
- [ ] Cut 1D manager: `cut_1d`, `cut_constant_q` (was `cut1DE`; `width` is now the
      diameter, twice the old value), `plot_cut_1d` and `plot_cut_constant_q` (was
      `plotCut1DE`) on a stored cut (`Plotting.plot_cut_...(result)`), ufit export
      (`Gui1DCutObject`: `ds.generate_ufit_dataset(result)` in place of
      `generateUFitDataset(pdData, q1, q2, ...)`)
- [ ] Q-E line manager: `plot_cut_qe`, `plot_cut_qe_line` (was `plotCutQELine`; returns the
      axes only, cuts in `ax.result`), `calculatePosition` on the axes
- [ ] Q plane manager: `plot_q_plane`; `DraggableShapes.extractCut` on the new axes; energy
      range from `gather`
- [ ] View 3D manager on `ds.view_3d`: pass one `on_cut(kind, parameters)` that opens drawn
      cuts in the GUI's cut managers (replaces the ten `cut1DFunction...`/`cut2DFunction...`
      arguments of `View3D`); embed `viewer.figure` in the Qt window; slice saving through
      `save_slices` (replaces `saveToFile(..., gui=...)`)
- [ ] If kept, the PyQtGraph free-plane slicer (`customSlicer=True`) lives in the GUI, on a
      `bin_3d` result, ported to `QtWidgets`
- [ ] Normalisation manager: `ds.absolute_normalize(sample_mass, sample_chemical_formula or
      sample_molar_mass, units='magnetic' or 'nuclear', ...)`; the sample mass and formula
      (or molar mass) are now required, so the dialog must ask for them
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
- [x] Read converted `.nxs` (NXsqom) files: Dropped (2026-10-19)
- [x] McStas simulated files: dropped for now (decided 2026-10-19), not ported from
      `DataFile`. To be added later as `McStasReader.py` (see "Later")
- [x] Copy a reader (old copy constructor `DataFile(other)`, used by subtraction): shallow
      copies in `__sub__`
- [x] `createEmptyDataFile`: simulated file for planning/resolution. Dropped (2026-10-19)
- [x] `shallowRead`: quick read of a few parameters (GUI log book): `shallow_read(path,
      keys)` and `shallow_keys()` on every reader, `DataReaderFactory.shallow_read` for many
      files. Nexus readers read only the datasets asked for, through the layout (every
      layout field is a key, plus `name`, `fileLocation`, `twoTheta`); NICOS readers read
      the file. Identical to the old `shallowRead` for 22 keys on all 20 CAMEA files, and to
      the reader's values and metadata
- [x] Clear error for files without data: `DataReader._check_has_data`, called by the Nexus
      readers on the counts as stored and by the NICOS readers on the scan table, raises
      `ValueError: "<file>" has no data (0 scan steps): ...` for no scan steps or a scalar.
      Before, CAMEA and Flatcone files with no steps loaded and failed later on a monitor
      mismatch; scalar counts and NICOS files without rows raised `IndexError`. Tests in
      `test_ReaderValidation.py`. A file with no counts dataset at all still fails
      `can_read` ("is not a CAMEA data file")

### CAMEA data not read yet
- [x] `MonitorMode`: `reader._metadata['mode']` (e.g. 'monitor'), with `title`, `comment`,
      `startTime`, `endTime` and the other `CameaLayout.metadata` entries
- [ ] `MonitorPreset` (`control/preset`, one value followed by zeros in NICOS files): only
      the 2018 monitor fix would use it (questions 9 and 11)
- [ ] Monitor fixes: 2018 wrong monitor, extra value in files of running scans
- [x] `Time` (counting time per step): `reader.time` [s], as the Marmot and NICOS readers;
      identical to the old `DataFile` for all 20 CAMEA files. `totalCounts`: Dropped
      (2026-10-19)
- [ ] `absoluteTime`: only the temperature-from-log fallback would use it (question 41)
- [x] `temperature`, `magneticField`, `electricField`: `SampleEnvironment` (measured and
      setpoints, units from the file), identical to the old DataFile for all CAMEA test files
- [ ] Temperature from the NICOS log when no temperature is stored
- [x] Scan parameters (`getScanParameter`: parameters, values, units):
      `CameaDataReader.get_scan_parameters` -> `scan_parameters`, `scan_values`, `scan_units`
      (as the other readers): variables in `entry/data`, else named in `entry/scanvars`
      (`a3` -> A3, `s2t`/`a4` -> A4, `ei` -> Ei), else for a manual scan the first of
      temperature, field, Ei, A3, 2theta that changes. Identical to the old `DataFile` for all
      20 CAMEA files. Not ported: Q scans (`qcscan`, scan variable `CAMEA`; no test data):
      left out with a warning, as are unknown variables (the old code added the name
      without values)
- [x] `scanCommand` (the GUI shows it): `reader._metadata['scanCommand']`, as the NICOS
      readers (`CameaLayout` lists it under metadata now, not fields); identical to the old
      `DataFile` for all 20 CAMEA files
- [x] `protonBeam`, `singleDetector1/8`, `s2tRaw`, analyser / monochromator settings,
      `totalCounts`: Dropped (2026-10-19) as reader attributes; they stay in `CameaLayout`, so
      `shallow_read` and `getValue` still read them

### Conversion output
- [x] Monitor spread to the shape of the data
- [x] Default mask (nothing masked; reset by `transform()`)
- [x] CAMEA binning 8: mask the first 2 pixels
- [x] CAMEA: mask broken tubes (NaN calibration A4), with a message (general rule for all
      instruments: any pixel with a NaN A4, Ef or amplitude)
- [x] Marmot: mask Ef < 1
- [x] Normalization spread over all scan steps
- [x] Detector/analyser selection (`dasel`) and the index calculation from it:
      `detector_selection`, `analyser_selection` on every reader (read from CAMEA files,
      else 0, 0, as the old `DataFile`); `DataReader.raw_counts(detector, analyser)` replaces
      `calcualteDataIndexFromDasel`: the analyser's pixels at binning 1 (calibration
      updates included) summed, or the channel for instruments with one channel per
      analyser (NICOS, Flatcone, Marmot). Leaves the reader's binning and calibration as
      they were
- [x] `DataSet` default binning = the reader type's default binning

### Calibration and sample
- [x] `updateCalibration`: external normalisation tables, several binnings, `overwrite`.
      `DataReader.update_calibration(files, overwrite=False)` reads `.calib` files
      (`read_calibration_file`, sharing `calibration_from_table` with the CSV tables of the
      other readers) into `calibration_updates`, used by `set_instrument_calibration` in
      place of the file's; new binnings join `possible_binnings`; the row count is checked.
      `DataSet.update_calibration` applies them to every file, also files added later, and
      transforms again if the current binning changed. `CameaDataWriter` saves the updated
      calibrations by default. Calibrations and converted data (I, normalization, monitor,
      energy, qx, qy) identical to the old code with the 2024 tables in `Data/`, binnings 1,
      3, 5 and 8 / 8 and 3. Fixed: the old method took a single path as a list of characters
- [x] Calibration plots: `plotA4`, `plotEf`, `plotEfOverview`, `plotNormalization` as
      `Plotting.plot_calibration_a4`, `plot_calibration_ef`, `plot_calibration_ef_overview`,
      `plot_calibration_normalization`; each takes an `InstrumentCalibration`
      (`reader.get_instrument_calibration(binning)`) and `n_detectors`, an optional `ax`, and
      returns the axes. Normalisation = amp * width * sqrt(2 pi)
      (`Plotting.calibration_normalization`), as before. Old ones failed on NICOS-written files
      (`DataFile` has no `detectors` for them)
- [x] `updateSampleParameters`: `reader.update_sample_parameters(unit_cell)` keeps U, sets
      UB = U B(new cell) and converts H, K, L of transformed data from Qx, Qy again (the old
      method left them until the next conversion). Matches the old code after conversion
- [x] `calculateCurratAxeMask`: spurion masking, moved to `Mask.py` (works for every reader)

### Saving
- [x] `saveNXsqom`: save converted data. Dropped (2026-10-19) with converted files
- [x] `saveHDF`: write a raw HDF file. Not ported: replaced by `CameaDataWriter`, which keeps
      the whole file and writes only the calibration. With `update_calibration` this
      replaces the GUI's use: `ds.update_calibration(files, overwrite=True)`, then
      `CameaDataWriter(reader).save(new_path)` per file

### Utilities
- [ ] Compare files (`__eq__`, `difference`)
- [ ] `__str__`, `shape` / `size`
- [x] Lists of supported instruments and formats: `InstrumentName` (old
      `supportedInstruments`); `file_extensions` on each reader, all of them from
      `DataReaderFactory().file_extensions()` (old `supportedRawFormats`): CAMEA `hdf`,
      Flatcone and Marmot `nxs` (the old list missed it, as `.nxs` meant converted files),
      MultiFLEXX and BAMBUS `dat` or none

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
- [ ] Normalisation and calibration files as inputs. Converted files as inputs: Dropped
      (2026-10-19)
- [ ] `settings` dict and extra keyword arguments
- [x] Save converted files (`convertDataFile` with `saveFile`, `saveLocation`,
      `deleteOnConvert`): Dropped (2026-10-19) with converted files
- [x] `updateSampleParameters` for the whole set: `ds.update_sample_parameters(unit_cell)`.
      Kept for files added later (with the files' cell or the new one; the old set refused
      them as a different sample) and slices; masks from a mask object are evaluated again
      and a background model fitted again. Refused on a difference. Also done:
      `updateCalibration` (`DataSet.update_calibration`)

### Combined data
- [x] Drop masked points when extracting data (old `extractData()`), `maskIndices`
      (`gather`; boundaries are `view.offsets[1:]`)
- [x] `I_err`: readers have `intensity_error` (set by `ds1 - ds2`); every cut uses it
      (`cut_1d`, `cut_qe`, `cut_qe_line`, `cut_q_plane`, `bin_3d` since 2026-11-12)
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
- [x] `absoluteNormalize` / `undoAbsoluteNormalize`: `ds.absolute_normalize(sample_mass,
      sample_chemical_formula=None, sample_molar_mass=None, units='magnetic', **options)`
      (returns the factor), `ds.undo_absolute_normalize()`; the factor in
      `AbsoluteNormalization.absolute_normalization_factor` (snake_case arguments; the old
      `_tools.calculateAbsoluteNormalization` stays for the old code). Readers keep a
      `normalization_factor` that `transform` applies to the normalization from the
      calibration, so the factor survives `set_binning`, `update_calibration`, files added
      later, slices and differences; `ds.absolute_normalization` holds it. A second call
      replaces the first. The background model is refitted (cuts with background scale
      as a whole). Factor identical to the old one for every option; normalised cuts
      identical to the old `cut1D` after `absoluteNormalize`. Changes under "Cut bugs fixed
      in the port"
- [x] `symmetrize`: Dropped (2026-11-11). Used by nothing but an uncollected old test (the
      FAQ's "symmetrize" entry describes another technique: loading the files again with
      rotated A3 offsets). It overwrote qx, qy in place with no undo, was lost on
      reconversion, and dropped any out-of-plane component. Reasons in the refactoring log

### Coordinates
- [x] `convertToQxQy`, `convertToHKL` (`convert_to_qxqy`, `convert_to_hkl`)

### Cuts
- [x] `cut1D` (`Cuts.cut_1d`; identical to the old code apart from the `Int_err` fix)
- [x] `cut1DE`: `Cuts.cut_constant_q`, `ds.cut_constant_q(q, width=0.04, min_pixel=0.1,
      energy_min, energy_max, rlu, constant_bins, background_subtraction)`. Identical to the
      old `cut1DE` (all columns, bin edges; adaptive and constant bins; 1-3 files, rlu on and
      off) with the old width doubled. Changes, listed under "Cut bugs fixed in the port":
      `width` is the diameter, files aligned with `rlu`, measurement errors used, `intensity`
      keeps the counts with background. Not ported: `ufit=True` (use
      `ds.generate_ufit_dataset(result)`), `dataFiles=`; an empty cut returns an empty
      result instead of raising; no `BinDistance` column (as `cut_1d`)
- [x] `cutQE` (`Cuts.cut_qe`; identical to the old code)
- [x] Measurement errors (`I_err`): every cut takes `intensity_error` (`cut_1d`, `cut_qe`,
      `cut_qe_line`, `cut_q_plane`, `bin_3d` added 2026-11-12), and the `DataSet` cuts pass
      the readers' errors (`ds1 - ds2`). Without them the results are unchanged
- [x] `cutQE` `smoothing` (a Gaussian filter of `Int` after binning): Dropped (2026-11-12).
      Display only; it returned smoothed `Int` with the unsmoothed errors. Smooth
      `result.grid('Int')` or the plotted image instead
- [x] `cutQE` `scaledEnergy` (bins even in sqrt(|E|/f)): Dropped (2026-11-12). The scaling is
      monotonic, so the same bins come from passing `energy_bins` = `_tools.rescale(
      np.linspace(_tools.scale(E_min, f), _tools.scale(E_max, f), n), f)` to `cut_qe`
- [x] `cut1D` `plotCoverage` (debug figure of the points and the cut's box in Qx, Qy):
      Dropped (2026-11-12). The `CutResult` bins hold the box if a plot is wanted later
- [x] ufit output: no `ufit=True` option; `ds.generate_ufit_dataset(result)` exports any 1D
      cut
- [x] `cutQELine`: `Cuts.cut_qe_line`, `ds.cut_qe_line(q_points, energy_bins, width=0.1,
      min_pixel=0.01, rlu, background_subtraction)`: one `cut_qe` per segment, in one
      `CutResult` (`qCut`, `BinDistance` from the segment start; bins `q_points`, `offsets`,
      per-segment `distance` and `energy`) instead of lists. `energy_bins`, `width`,
      `min_pixel` shared or per segment. The old `constantBins` (default True) did nothing:
      `cutQE` only used it in a branch that never ran (`if False:`); not ported
- [x] `cutELine`: Dropped (2026-10-19)
- [x] `cutPowder`: `Cuts.cut_powder(..., q_bins, energy_bins, background, intensity_error)`,
      `ds.cut_powder(q_bins, energy_bins, background_subtraction)`. NOTE the argument order:
      |Q| bins first (the old `cutPowder(EBins, QBins)` took the energy bins first). A
      `CutResult` with the old columns (`q`, `Energy`, `Int_err`, `Int`, `intensity`,
      `Normalization`, `Monitor`, `BinCount`; `COLUMNS['qLength']` = `q`), bins `q` and
      `energy`, and `result.grid(column)` of shape (|Q| bins, energy bins). Every column
      identical to the old `cutPowder` without background. Uses measurement errors
      (`intensity_error`). Not ported: `constantBins` (never used), `dataFiles=`. Changes
      under "Cut bugs fixed in the port"
- [x] `binData3D` (`Cuts.bin_3d`, `ds.bin_3d`, `CutResult.grid`; `calculateBins` and
      `calculateGrid3D` reduced to `Cuts.grid_edges`, one edge array per axis). Identical
      to the old code. Not ported: `noApproximation` (returned 2 arrays instead of 4) and the
      debug branch `backgroundSubtraction == 2`
- [x] `cutRaw1D`: `ds.cut_raw_1d(detector=None, analyser=None)` -> one `CutResult` per
      file (columns per scan variable, `intensity` (counts), `Monitor`; parameters file,
      detector, analyser, scan variables and units) instead of `(X, I, parameter, unit)`.
      Detector and analyser shared or per file; files must scan the same variables; frozen
      data sets refused. Counts and scan values identical to the old `cutRaw1D` for all 20
      CAMEA files (file selection and three others) and all 100 Bambus channels
- [x] `extractData` (raw selection by A4 / Ef): Dropped (2026-10-19). Used only in the
      FAQ; CAMEA-only (104 x 8 hard-coded) and binning 1 only. `raw_counts(detector,
      analyser)` gives a channel's counts; `gather` the converted data
- [x] Pandas output columns (`pdNaming`): `CutResult.COLUMNS`

### Plotting and interactive figures
- [x] `plotCut1D`: `Plotting.plot_cut_1d(result, sample, rlu, ax, counts, show_energy,
      plot_foreground, plot_background, **errorbar options)` and `ds.plot_cut_1d(q1, q2,
      width, min_pixel, energy_min, energy_max, ..., **plot_options)`, on the existing cut
      axes (`_cut_axes`, the port of `generate1DAxis`). Returns the axes (`ax.Data`,
      `ax.result`, `ax.to_csv`) instead of `(ax, Data, bins)`; axes of an earlier 1D cut
      are reused, so cuts can share them. Drawn data, positions (`BinDistance`), labels,
      tick labels and limits identical to the old `plotCut1D` (rlu on and off, counts True /
      False / 'sensitivity', `show_energy`, background with foreground and background
      curves), except the `Int_err` bars, which no longer have the old truncation (up to
      2.8% here). Not ported: `ufit=True` (use `ds.generate_ufit_dataset(ax.result)`),
      `plotCoverage`, `dataFiles=`; `outputFunction` is `on_click`
- [x] `plotCut1DE`: `Plotting.plot_cut_constant_q(result, sample, rlu, ax, counts, show_q,
      plot_foreground, plot_background, **errorbar options)` and
      `ds.plot_cut_constant_q(q, width=0.04, min_pixel, energy_min, energy_max, ...,
      **plot_options)`: `Int` with `Int_err` against energy, the Q point in the tick labels
      (`show_q`). Shares the drawing of `plot_cut_1d` (`_draw_1d`). Returns the axes
      (`Data`, `result`, `startPoint`, `to_csv`, `type = 'ConstantQ'`); axes of an earlier
      constant-Q plot are reused. Drawn points, error bars, labels and tick labels identical
      to the old `plotCut1DE` (old width doubled) for rlu on and off, `show_q` off, counts
      True / 'sensitivity', and background with foreground and background curves. As in the
      old method, asking for those curves turns background subtraction on. The 3D viewer's
      default constant-Q cut uses it. Not ported: `ufit=True` (use
      `ds.generate_ufit_dataset(ax.result)`), `data=`, `dataFiles=`; the click printout is
      `on_click`
- [x] `plotCutQE` (`Plotting.plot_cut_qe`)
- [x] `plotCutQELine`: `Plotting.plot_cut_qe_line(result, sample, rlu, ax, counts, log,
      vmin, vmax, cmap, colorbar, ticks, plot_separator, separator_color, separator_width,
      grid)`, `ds.plot_cut_qe_line(...)`. Returns the axes (`pmeshs`, `set_clim`/`get_clim`,
      `QPoints`, `QPointsHKL`, `OffSets`, `calculatePosition`, `calculateIndex`, `Data` per
      segment, `result`, `to_csv`, `type = 'QELine'`). Mesh positions and values, offsets,
      labels, tick labels (also with `ticks`) and limits identical to the old plot; `counts`
      added as in `plot_cut_qe`. Changes under "Cut bugs fixed in the port". Not ported:
      `dataList=` (pass a `cut_qe_line` result to `Plotting.plot_cut_qe_line`), `dataFiles=`,
      `_interactiveSettings` modes; the click printout (`outputFunction`) is `on_click`
- [x] `plotCutELine`: Dropped (2026-10-19)
- [x] `plotCutPowder`: `Plotting.plot_cut_powder(result, ax, counts, log, vmin, vmax, cmap,
      colorbar=True, grid, **pcolormesh options)`, `ds.plot_cut_powder(q_bins, energy_bins,
      background_subtraction, **plot_options)`. Returns the axes (`pmeshs`, `set_clim`,
      `get_clim`, `calculateIndex`, `Data`, `result`, `to_csv`, `type = 'Powder'`). Plotted
      values, colour limits and labels identical to the old plot. `counts` added as in the
      other maps. Click printout: `on_click`. Not ported: `dataFiles=`
- [x] `plotQPlane` (`Cuts.cut_q_plane`, `Plotting.plot_q_plane`)
- [x] `plotQPlane` on 3D axes (each energy range a `contourf3D` layer at its mean energy):
      Dropped (2026-11-12). Unused by the GUI, the docs and the tutorials; the read-out,
      cutting and lattice grid did not work on it; `view_3d` (`SliceViewer`) moves through
      the energy slices. `plot_q_plane` refuses 3D axes (it would draw every range in one
      plane)
- [x] `plotQPlane` `binning='polar'` (bins in angle and |Q|), `enlargen` (adaptive bins) and
      `scaleFunction` (a function applied to the intensity): Dropped (2026-11-12). Unused by
      the GUI and the docs, and broken: `polar` with `enlargen` always raised (`binEdges(x,
      tolrance=...)`); bin edges were stored only for energy ranges with points but looked up
      by the range's index, so an empty range gave an `IndexError` or shifted the bins of
      later ranges; `scaleFunction` ignored the background (`if not backgroundSubtraction is
      None` is always true). Use `log=True`, or rescale `ax.Int` / the result's `Int`
- [x] `plotRaw1D`: `Plotting.plot_raw_1d(results, ax, legend, grid, **scatter options)`,
      `ds.plot_raw_1d(detector, analyser, **plot_options)`: a scatter per file of counts
      against the first scan variable, legend of file names (or given names), read-out of
      all scan variables at a position (interpolated) and the counts, `ax.results`,
      `ax.to_csv`. Returns the axes. The click printout (`outputFunction`) is `on_click`
- [x] `plotA3A4` (with `boundaryQ`, `convexHullPoints`): Dropped (2026-10-19)
- [x] `View3D`: `ds.view_3d(dx, dy, dz, rlu, background_subtraction, **viewer options)`,
      `SliceViewer.SliceViewer(result of bin_3d, sample, rlu, axis, counts, log, vmin, vmax,
      cmap, grid, adjustable, on_cut, dataset, cut_width, cut_min_pixel, spurion_peaks, Ei,
      ef_limits)`. Plain matplotlib: slices along Qx, Qy or energy (constant-energy slices on
      `RLUAxes.createQAxis` with rlu; Q-E slices on plain axes, tick labels in H, K, L), a
      slider, keys (arrows, scroll, 0/1/2, c for cutting), colour sliders, `set_clim`,
      read-out of H, K, L, E and value, Currat-Axe markers, `to_csv` (current slice),
      `save_slices`. Cuts drawn: drag on an energy slice -> '1d' (`cut_1d`), click on a Q-E
      slice -> 'constant_q' (`cut_constant_q`), drag on a Q-E slice -> 'qe' (`cut_qe`), passed
      to `on_cut(kind, parameters)` or plotted by the data set (`SliceViewer.default_cut`).
      Slice values and edges identical to the old `Viewer3D` (rlu on and off); H, K, L of
      positions equal the readers' H, K, L; elastic spurions on their Bragg peaks. Changes
      under "Cut bugs fixed in the port". Not ported: `instrumentAngles`, `grid` as a zorder,
      `customSlicer` (see the 3D viewer split), the old `_interactiveSettings` modes
- [x] `createQAxis` / `createQEAxes`: `ds.create_q_axis(rlu, figure, ids, basex, basey)` and
      `ds.create_qe_axes(axis, figure, projection_vector1, projection_vector2, rlu,
      figure_position)`, wrapping `RLUAxes` (which reads the sample from a list, so it failed
      on the new `DataSet`). Labels and read-out identical to the old methods (rlu on and
      off, both Q-E axes). `create_q_axis` has no click handler (the old default one failed
      on empty axes: it read the bins of a Q-plane plot); `plot_q_plane(ax=...)` adds its own
- [x] `generate1DAxis(E)`: inside `plot_cut_1d` / `plot_cut_qe` (`_cut_axes`) and
      `plot_cut_constant_q`; not public (no users outside the old code)
- [x] Click read-out (the old `outputFunction`): every plot (`plot_cut_1d`,
      `plot_cut_constant_q`, `plot_cut_qe`, `plot_cut_qe_line`, `plot_cut_powder`,
      `plot_q_plane`, `plot_raw_1d`) and `SliceViewer` take `on_click` (default `print`; None
      off): a click, not while zooming or panning, passes the position and the bin's Int,
      Cts, Norm, Mon, NormCount (`Plotting.bin_text`). `ax.read_out(x, y)` and
      `ax.bin_at(x, y)` (the data row) without clicking. The old check for the Qt cursor
      shape is replaced by the toolbar mode; on a Q plane with overlapping energy ranges the
      read-out is of the last range with points there (the old one always used the first);
      counts of derived data are not truncated to integers
- [x] CSV export and hover: done. The `_interactiveSettings` modes (drawing cuts on plots):
      moved to Milestone 3 (2026-11-12)

### Instrument physics
- [x] `calculateResolutionMatrix`, `calculateResolutionMatrixAndVectors`:
      `ds.resolution_matrix(position, Ei, Ef, rlu)` and `ds.resolution_ellipse(position,
      direction1, direction2, Ei, Ef, rlu, lattice_axes)` (was `rluAxis`), on
      `MJOLNIR/Data/Resolution.py` (`camea_resolution_matrix(qx, qy, Ei, Ef, analyser,
      a3_offset)`, `project`; the Eckold-Sobolev calculation of `Geometry/eck_Flipped.py`,
      imported without `Instrument.py` and its Qt import). The analyser is the one whose
      calibrated Ef, in the first file's own calibration, is nearest to Ef (the old one read
      the default tables shipped with the package). CAMEA only: other instruments raise
      `NotImplementedError`. Matrices identical to the old code at binning 8 (within 1e-12,
      three positions x one Ef per analyser), ellipse axes and widths too, also on lattice
      axes; Qx, Qy input. Bugs fixed: listed under "Bugs in the old code to fix, not port".
      Tests: `test_Resolution.py`, `test_Reference.py`. The ellipses drawn on plots stay with
      the interactive modes (Milestone 3)
- [x] `findEi`: `ds.find_ei(energy_transfer)`, the incoming energies at which an energy
      transfer is measured. Uses every Ei of a file (the old one its first) and ignores
      pixels without a calibrated Ef (NaN). Identical to the old code on camea_test (six
      incoming energies, 11 energy transfers), within single precision (the old Ei was float32)
- [x] `checkSecondOrderElastic`: `ds.second_order_elastic(energies, atol=0.2)`, the Ei whose
      elastic line the analysers pass in second order at each energy (4 Ef = Ei), NaN where
      none; no printing. Identical to the old code on camea_test (five peaks, three
      tolerances), within single precision. No test file reaches the line itself (it needs
      Ei >= 12.8 meV, as CAMEA's lowest Ef is 3.2 meV)

### Export and saving
- [x] `generateUFitDataset` (ufit): `Export.to_ufit(result, rlu, meta)` and
      `ds.generate_ufit_dataset(result, rlu)` (metadata: instrument, experiment, title, file
      names) take a `CutResult` of `cut_1d` (x along Q) or `cut_constant_q` (x = energy)
      instead of a DataFrame plus the cut's arguments; other results are refused. ufit is
      imported only when exporting. Labels (`xcol`), y and errors identical to the old export
      for cuts along Q (rlu on and off), x and y along energy. Changes under "Cut bugs fixed
      in the port"; `datafilename` holds the raw file names (the old one the converted
      `.nxs` names)
- [x] `save` / `load`: Dropped (2026-10-19)

## Bugs in the old code to fix, not port
- [x] Resolution matrix (`Instrument.calculateResoultionMatrix`), fixed in
      `Data/Resolution.py`: the analyser (and so the sample-analyser distance) was the Ef bin
      index // 8, right only at binning 8 (the same point gave a Qx term of 7214 at binning 1,
      7628 at 2, 8940 at 8); a Qx, Qy position without a sample raised (`if rlu or sample is
      None` then used the sample); another instrument raised `UnboundLocalError` (a
      `NotImplementedError` built but not raised); the binning check let 9 and more through
      (`not binning>0 and binning <9`)
Recorded as expected failures in `test/test_DataSetSmoke.py`, except the `saveHDF` items,
which are resolved by not porting it: `CameaDataWriter` replaces `saveHDF`.
- [x] Combined `h/k/l` point at attributes `H/K/L`: Dropped (2026-10-19);
      `gather(ArrayType.H)` etc. replace them
- [x] `cutELine` parameter misspelt `EMn`; `plotCutELine` always fails: Dropped (2026-10-19)
- [x] `plotA3A4` calls the missing `voronoiTessellation`: Dropped (2026-10-19)
- [x] `removeMasking` fails unless a `Mask` object was applied (new `remove_masking` works)
- [x] `save` cannot pickle the data set: Dropped (2026-10-19)
- [x] `saveHDF` crashes on every NICOS-written CAMEA file (2023 and 2024 data): sample
      `x`/`y` are absent, read as `array(None, dtype=object)`, and h5py cannot store them
- [x] `saveHDF` opens its target with mode `'w'`, so the crash leaves a truncated file; the
      GUI calibration update writes over the original raw file (`d.saveHDF(d.fileLocation)`),
      destroying it (1.6 MB file left as a 0.56 MB stub without counts, checked on a copy)
- [x] `saveHDF` rebuilds the file from what `DataFile` read, in the pre-NICOS layout: drops
      `detector/data`, `total_counts`, `monitor_2`, the temperature and field logs and
      `qh/qk/ql`; adds fields the source lacks; `CameaDataReader` cannot read the result
- [x] `Sample.updateSampleParameters` kept the old B, so a second update without a
      conversion in between took U from the wrong B and turned the crystal (UB off by 0.017
      for a 2% then 4% change of a). Fixed in `Sample.py`: it now stores the new B

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
- [x] `cut1DE` (now `cut_constant_q`): `width` was used as the radius of the disc around q,
      while its docstring and `cut1D` use it as the full width; now the diameter (old width w
      = new 2w; default 0.02 -> 0.04)
- [x] `cut1DE` with `rlu` used each file's own qx, qy (only q was converted, with the first
      file's UB): files with slightly different orientations were cut at different points.
      Now rotated into the first file's frame (`gather_q`), as `cut_1d` and `cut_qe`. No
      difference for the 2023 and 2024 test data (identical orientations)
- [x] Measurement errors: `cut1DE` never passed `I_err` to its binning function (so errors
      were always sqrt(counts)), and `cut1D` with `I_err` summed the squared errors without
      the square root, so `Int_err` was a variance in counts^2. Now readers carry
      `intensity_error` (None: counting statistics), `ds1 - ds2` sets it to
      sqrt(e1^2 + (e2 M1/M2)^2), and `cut_constant_q` sums the variances per bin and takes
      the root: `Int_err = sqrt(sum e^2) BinCount / (Monitor Normalization)`, with the
      background added to the variance when subtracted. Every other cut does the same
      (2026-11-12): on a difference of the sample files, `cut_1d` gave NaN `Int_err` in 74
      of 88 bins (the root of negative counts); now none
- [x] `cut1DE` with background: `intensity` held counts minus background (as `cutQE`); now
      the measured counts, as `cut1D`
- [x] `plotCut1D` with an unknown `counts` value built an `AttributeError` without raising
      it, and plotted nothing; `plot_cut_1d` raises
- [x] `plotCutQELine` colour scale: `vmin`/`vmax` were refused by its keyword check, and the
      common scale it computed was never applied, so every segment had its own scale (0-46,
      0-4.2 and 0-0.39 on the test path: one colour, intensities 100x apart). Now one scale
      for all segments, from `vmin`/`vmax` or the minimum and maximum of all segments
- [x] `plotCutQELine` with `rlu=False` failed: the Q points were converted from H, K, L
      regardless; now taken as Qx, Qy
- [x] `plotCutQELine` separators were drawn at offset minus a width that was always 0
      (`OffSetWidth`); dropped, the lines are at the Q points
- [x] `plotCutQELine` on 3D axes drew nothing; `plot_cut_qe_line` raises. `ticks=0` set
      ticks anyway (after a warning); now no ticks
- [x] `plotCutQELine` returned `(ax, dataList)`, not the documented `(ax, DataList,
      BinList)`
- [x] `View3D(counts=True)` showed counts divided by the number of points in the voxel,
      though documented as the number of neutrons; `SliceViewer(counts=True)` shows the
      counts in the voxel
- [x] `View3D(instrumentAngles=True)` computed A3, A4 with a fixed Ei = Ef = 4.95 meV
      whatever the data; not ported (needs the data's Ei and the slice's Ef to be right)
- [x] `generateUFitDataset` along Q measured x as the unsigned distance from the second bin,
      so the first bin came after the second (0.0117, 0, 0.0123, ...); now the signed
      distance from the first bin, increasing along the cut
- [x] `generateUFitDataset` computed the errors from the counts alone, ignoring a subtracted
      background (and measurement errors); now `Int_err` of the cut
- [x] `generateUFitDataset` along energy labelled x with the Q point and a mean energy as if
      it were a Q cut; now 'E [meV] at (H, K, L) [RLU]'
- [x] `cutRaw1D` with a single detector or analyser number gave counts for the first file only
      (the number became a one-element list, zipped with the files); `cut_raw_1d` uses it for
      every file
- [x] `cutRaw1D` / `plotRaw1D` failed on Flatcone files (IndexError: counts indexed as 3D);
      `cut_raw_1d` works for every reader
- [x] `calcualteDataIndexFromDasel` checked the analyser against Bambus's limit for every
      non-CAMEA instrument, hard-coded CAMEA's 8 analysers and accepted negative numbers;
      `raw_counts` checks 0 <= detector < n_detectors and 0 <= analyser < n_analysers
- [x] `plotRaw1D` read-out chose decimals from the natural logarithm of the step; now log10
- [x] `plotCut1DE` with an unknown `counts` value built an `AttributeError` without raising
      it; given axes got no labels or tick formatting; its read-out formatted L with
      `{:-3e}` (a missing dot). `plot_cut_constant_q` raises, sets up given axes, and
      formats L like H and K
- [x] `cutPowder` with background: `intensity` held counts minus background and
      `Int_Fg`/`Int_Bg` raw counts (as the old `cutQE`), and `Int_err` ignored the background
      (`sqrt(Int + BGInt)` taken after `Int` had the background subtracted, i.e.
      sqrt(counts)). Now the columns of every other cut, `Int_err` from counts and background
- [x] `plotCutPowder`: `ax.get_clim()` raised (`ax.pmesh[0]` on a single mesh); `log=True`
      added its offset after the logarithm (`log10(Int) + 1e-20`), so zero intensity gave
      -inf; bins were drawn around their centres (`shading='auto'`), misplacing the
      boundaries of unequal bins. Now `get_clim` works, `log10(value + 1e-20)`, and bins are
      drawn between their edges. The docstrings promised qx, qy, H, K, L columns
- [x] `undoAbsoluteNormalize` multiplied the normalization by the factor again instead of
      dividing (normalization factor^2 of the original, flags saying "not normalised");
      `undo_absolute_normalize` restores the normalization exactly
- [x] `absoluteNormalize` scaled the converted files in place, so converting again dropped
      it while the data set still recorded it as applied (a later call then applied only
      the ratio); now kept through every transform
- [x] `absoluteNormalize` without a sample mass still scaled by 16 pi/13.77 = 3.65 (mass and
      molar mass set to 1, g = 2), though commented "No normalization to sample!"; now the
      sample mass and formula or molar mass are required (decided 2026-10-19)
- [x] `absoluteNormalize` always converted to magnetic units (g^2/13.77); `units='nuclear'`
      gives barn per formula unit (4 pi m V / M). Its docstring listed `sampleDebyeWaller`
      and `sampleFormFactor`, which were not parameters

## Code still using the old API
- [ ] `Mask.py`: imports switched to `DataReader` / new `DataSet` (masks on `qx/qy`,
      `h/k/l`, `energy`, `A3` work); metaclass replaced by `ABC`; coordinates accept
      `ArrayType` or strings. `CurratAxeMask` works on readers; masks apply through
      `DataSet.mask`. Still to do: the bugs listed under "Mask.py bugs"
- [x] `BackgroundModel.py`: the powder model is the new `BackgroundModel.py`; the old file
      is kept as `BackgroundModel_old.py` for `DataSet_old`
- [ ] `Viewer3D.py` (deleted with the old code), `RLUAxes.py`. Moved to Milestone 3
      (2026-11-12): `Viewer1D.py`, `DraggableShapes.py`, `_interactiveSettings.py`
- [ ] `Statistics/CutObject.py`
- [ ] `_tools.py` (`PointerArray`)
- [x] Command line scripts: 3DView, History, CalibrationInspector. Convert: Dropped
- [x] Command line scripts take an `--instrument` option instead of detecting it

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
- [ ] Resolution matrices for the other instruments (Bambus, MultiFLEXX, Flatcone, Marmot):
      `ds.resolution_matrix` is CAMEA only. The Eckold-Sobolev calculation
      (`Geometry/eck_Flipped.py`) takes any three-axis geometry, so each needs its distances,
      component sizes, focusing, collimation and mosaics (`Data/Resolution.py` holds CAMEA's),
      and the multi-analyser instruments the analyser for each Ef. Decide which are needed and
      where the instrument parameters come from (constants per instrument, or the files)
- [ ] McStas simulated files: `MJOLNIR/Data/McStasReader.py`, a `McStasReader` derived from
      `DataReader`, registered in `DataReaderFactory`. Written new rather than ported: the
      old `DataFile` code can serve as a reference for the file layout until it is deleted
      (Milestone 2), and from git history after that

## Finish
Removing `DataFile.py` and `DataSet_old.py` is Milestone 2.
