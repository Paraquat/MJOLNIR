.. _Migration:

Migrating from MJOLNIR 1.x
==========================

MJOLNIR 2 replaces the old ``DataFile`` and ``DataSet`` with data readers and a new ``DataSet``. Scripts written for MJOLNIR 1.x need changes: the methods have new names (in snake_case) and some new arguments, cuts return a ``CutResult`` and plots return the axes. The data themselves are unchanged: the cuts give the same values as before, apart from the bugs fixed (listed below).

MJOLNIR 2 needs Python 3.11 or newer and numpy 2.3 or newer.


What changed
------------

- **No conversion step.** A data file is read by the data reader of its instrument and converted into reciprocal space when it is added to a DataSet. ``convertDataFile`` and the converted ``.nxs`` files are gone; the binning is set with ``ds.set_binning``.
- **The DataSet knows its instrument.** ``DataSet(InstrumentName.CAMEA, files=...)`` reads the files with the CAMEA reader; all files of a DataSet are of one instrument and one sample.
- **Cuts return a CutResult.** The binned data are a pandas DataFrame in ``result.data``, with the same columns as before (``Qx``, ``Qy``, ``H``, ``K``, ``L``, ``Energy``, ``intensity``, ``Monitor``, ``Normalization``, ``BinCount``, ``Int``, ``Int_err``); the bin edges are in ``result.bins`` and the cut's arguments in ``result.parameters``.
- **Plots return the axes.** The data are on the axes: ``ax.result`` (the CutResult) and ``ax.Data`` (the plotted DataFrame); ``ax.to_csv(fileName)`` saves them.
- **Values of all files.** ``ds.gather('qx')`` gives the unmasked values of all files (the old ``ds.qx.extractData()``); iterating over a DataSet gives its readers.


Loading data and making a cut
-----------------------------

Before (1.x):

.. code-block:: python

    from MJOLNIR.Data import DataSet
    from MJOLNIR import _tools

    files = _tools.fileListGenerator('1486-1487', '/path/to/data/', 2024)
    ds = DataSet.DataSet(dataFiles=files)
    ds.convertDataFile(binning=8)

    data, bins = ds.cut1D(q1=[1, 0, 0], q2=[1, 1, 0], width=0.1, minPixel=0.02,
                          EMin=0.0, EMax=1.0)
    ax, data, bins = ds.plotCut1D(q1=[1, 0, 0], q2=[1, 1, 0], width=0.1, minPixel=0.02,
                                  EMin=0.0, EMax=1.0)

After (2.x):

.. code-block:: python

    from MJOLNIR.Data.DataSet import DataSet
    from MJOLNIR.Data.InstrumentName import InstrumentName
    from MJOLNIR import _tools

    files = _tools.fileListGenerator('1486-1487', '/path/to/data/', 2024)
    ds = DataSet(InstrumentName.CAMEA, files=files)
    ds.set_binning(8) # Only needed for another binning than the instrument's default

    result = ds.cut_1d(q1=[1, 0, 0], q2=[1, 1, 0], width=0.1, min_pixel=0.02,
                       energy_min=0.0, energy_max=1.0)
    data = result.data
    ax = ds.plot_cut_1d(q1=[1, 0, 0], q2=[1, 1, 0], width=0.1, min_pixel=0.02,
                        energy_min=0.0, energy_max=1.0)

The arguments are renamed the same way in every method: ``minPixel`` → ``min_pixel``, ``EMin``/``EMax`` (or ``E1``/``E2``) → ``energy_min``/``energy_max``, ``EnergyBins``/``EBins`` → ``energy_bins``, ``dE`` → ``energy_step``, ``constantBins`` → ``constant_bins``, ``backgroundSubtraction`` → ``background_subtraction``, ``showEnergy`` → ``show_energy``, ``outputFunction`` → ``on_click``.


Data sets
---------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 1.x
     - 2.x
   * - ``DataSet(dataFiles=files)``
     - ``DataSet(InstrumentName.CAMEA, files=files)``
   * - ``ds.convertDataFile(binning=3)``
     - ``ds.set_binning(3)`` (files are converted when added)
   * - ``DataFile.DataFile(path)``
     - ``DataReaderFactory().create(InstrumentName.CAMEA, Path(path))``
   * - ``DataFile.shallowRead(files, parameters)``
     - ``DataReaderFactory().shallow_read(instrument, files, keys)``: one dict per file
   * - ``ds.qx.extractData()``, ``ds.I.extractData()``
     - ``ds.gather('qx')``, ``ds.gather('intensity')`` (``np.asarray`` joins the files)
   * - ``ds.I``, ``ds.Monitor``, ``ds.Norm``
     - ``'intensity'``, ``'monitor'``, ``'normalization'`` (``gather``, ``get_view``)
   * - ``ds.sample[0]``
     - ``ds.sample`` (the first file's), ``ds.samples``
   * - ``d.A3Off`` of a data file
     - ``reader.A3Offset``, set before the reader is added to a DataSet
   * - ``d.temperature``, ``d.magneticField``
     - ``reader.environment.temperature``, ``reader.environment.magnetic_field``
   * - ``d.scanParameters``
     - ``reader.scan_parameters``, ``scan_values``, ``scan_units``
   * - ``ds.updateCalibration(files, overwrite)``
     - ``ds.update_calibration(files, overwrite)``
   * - ``ds.updateSampleParameters(cell)``
     - ``ds.update_sample_parameters(cell)``
   * - ``ds.saveHDF(...)`` (``DataFile``)
     - ``CameaDataWriter(reader).save(path)``, to a new file
   * - ``ds.autoSort()``, ``argAutoSort``, ``argSortTo``
     - ``ds.auto_sort()``, ``arg_auto_sort``, ``arg_sort_to``, ``apply_sorting``
   * - ``ds.convertToQxQy(hkl)``, ``convertToHKL(qxqy)``
     - ``ds.convert_to_qxqy(hkl)``, ``ds.convert_to_hkl(qxqy)``
   * - ``ds1 - ds2``
     - ``ds1 - ds2``, scaled by the measured monitor per point (was the preset)


Cuts and plots
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 1.x
     - 2.x
   * - ``cut1D``, ``plotCut1D``
     - ``cut_1d``, ``plot_cut_1d``
   * - ``cut1DE(E1, E2, q, width=0.02)``, ``plotCut1DE``
     - ``cut_constant_q(q, width=0.04, energy_min, energy_max)``, ``plot_cut_constant_q``: **width is the diameter**, twice the old value
   * - ``cutQE``, ``plotCutQE``
     - ``cut_qe``, ``plot_cut_qe``
   * - ``cutQELine(QPoints, EnergyBins)``, ``plotCutQELine``
     - ``cut_qe_line(q_points, energy_bins)``, ``plot_cut_qe_line``: one CutResult for all segments
   * - ``cutPowder(EBins, QBins)``, ``plotCutPowder``
     - ``cut_powder(q_bins, energy_bins)``, ``plot_cut_powder``: **Q bins first**
   * - ``plotQPlane(EMin, EMax, EBins)``
     - ``cut_q_plane``, ``plot_q_plane(energy_min, energy_max, energy_bins)``
   * - ``binData3D(dx, dy, dz)``
     - ``bin_3d(dx, dy, dz)``; ``result.grid('Int')`` gives the 3D array
   * - ``View3D(dQx, dQy, dE)``
     - ``view_3d(dx, dy, dz)``: the SliceViewer
   * - ``cutRaw1D(detectorSelection, analyzerSelection)``
     - ``cut_raw_1d(detector, analyser)``: one CutResult per file
   * - ``plotRaw1D``
     - ``plot_raw_1d``
   * - ``createQAxis``, ``createQEAxes``
     - ``create_q_axis``, ``create_qe_axes``
   * - ``generateUFitDataset(pdData, q1, q2, ...)``, ``ufit=True``
     - ``generate_ufit_dataset(result)``

Cuts that returned ``(data, bins)`` return a CutResult; plots that returned ``(ax, data, bins)`` return ``ax``. ``result.parameters`` records the arguments the cut was binned with, positions in Qx, Qy. An empty cut returns an empty result instead of raising.


Masks, background and normalisation
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 1.x
     - 2.x
   * - ``ds.mask = [...]``
     - ``ds.mask = [...]`` (one array per file), or a mask object of ``Mask``
   * - ``ds.appendMask(mask)``, ``ds.removeMasking()``
     - ``ds.append_mask(mask)``, ``ds.remove_masking()``
   * - ``ds.calculateCurratAxeMask(BraggPeaks, ...)``
     - ``ds.mask = Mask.CurratAxeMask(braggPeaks, ...)``
   * - ``ds.generateBackgroundModel(backgroundMask, dQ, dE)``
     - ``ds.generate_background_model(background_mask, dQ, dE)``
   * - ``AMBERBackground``
     - ``ds.generate_amber_background(...)``
   * - ``ds.absoluteNormalize(sampleMass, sampleChemicalFormula, ...)``
     - ``ds.absolute_normalize(sample_mass, sample_chemical_formula, ...)``: the sample mass and the formula (or molar mass) are required; ``units='magnetic'`` or ``'nuclear'``
   * - ``ds.undoAbsoluteNormalize()``
     - ``ds.undo_absolute_normalize()``
   * - ``ds.calculateResolutionMatrix(position, Ei, Ef)``
     - ``ds.resolution_matrix(position, Ei, Ef)``
   * - ``ds.calculateResolutionMatrixAndVectors(..., rluAxis)``
     - ``ds.resolution_ellipse(..., lattice_axes)``
   * - ``ds.findEi(deltaE)``, ``ds.checkSecondOrderElastic(EPeaks)``
     - ``ds.find_ei(energy_transfer)``, ``ds.second_order_elastic(energies)``

Masks now follow the binning: a mask object is evaluated again when the binning changes or files are added, and each file also has a default mask of the points its instrument cannot use (``ds.use_default_mask``).


Results that change
-------------------

The following bugs of 1.x are fixed, so results can differ from those of old scripts:

- ``cut_constant_q`` uses ``width`` as the diameter of the disc around q (``cut1DE`` used it as the radius: double old widths), rotates the files into the first file's frame with ``rlu`` (as ``cut_1d``), and uses the measurement errors of derived data.
- ``Int_err`` of 1D cuts is no longer truncated to whole counts.
- With background subtraction, ``intensity`` keeps the measured counts (it held counts minus background in ``cutQE``, ``cut1DE`` and ``cutPowder``).
- ``ds1 - ds2`` scales by the measured monitor of each point rather than the preset.
- ``absolute_normalize`` requires the sample mass (the old one scaled by a constant without it), supports nuclear units, and ``undo_absolute_normalize`` really undoes it.
- ``view_3d(counts=True)`` shows the counts per voxel (the old one divided them by the number of points), and ``plot_cut_qe`` honours ``counts``.
- ``generate_ufit_dataset`` takes the background into account in the errors and labels energy cuts correctly.

The refactoring checklist (``REFACTOR_CHECKLIST.md`` in the repository, "Cut bugs fixed in the port") has the full list.


Removed features
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Removed
     - Instead
   * - Converted ``.nxs`` files, ``saveNXsqom``, ``MJOLNIRConvert``
     - Converted in memory; ``ax.to_csv`` or ``gather`` to export
   * - ``DataSet(normalizationfiles=..., calibrationfiles=...)``
     - ``ds.update_calibration(files)``
   * - ``extractData`` (raw selection by A4 and Ef)
     - ``reader.raw_counts(detector, analyser)``, or ``gather``
   * - ``cutELine``, ``plotCutELine``, ``plotA3A4``
     - None
   * - ``symmetrize``
     - Read the files again with rotated ``A3Offset`` (see the FAQ)
   * - ``cutQE(smoothing=...)``, ``cutQE(scaledEnergy=...)``
     - Smooth ``result.grid('Int')``; pass scaled ``energy_bins``
   * - ``plotQPlane`` on 3D axes, ``binning='polar'``, ``enlargen``
     - ``view_3d`` for the energy slices; ``log=True`` for scaling
   * - ``View3D(customSlicer=True)`` (PyQtGraph viewer)
     - ``view_3d``; the PyQtGraph viewer moves to MJOLNIR-GUI
   * - ``DataSet.save`` / ``load``, ``settings``
     - Keep the file list and arguments in the script
   * - ``Viewer1D``, ``Statistics.FittingFunction``, ``CutObject``
     - Fit with ufit (``generate_ufit_dataset``) or scipy
   * - The interactive cutting modes on Q-plane and Q-E plots
     - Cutting in ``view_3d``; see `Interactivity <InDepthDocumentation/Interactivity.html>`_


Command line scripts
--------------------

``MJOLNIR3DView``, ``MJOLNIRHistory`` and ``MJOLNIRCalibrationInspector`` keep their arguments and have a new ``-i/--instrument`` option for files of other instruments than CAMEA (the default). ``MJOLNIRConvert`` is removed with the converted files. The scripts need PyQt5 for their file dialogue, which is not installed with MJOLNIR.
