.. :DataSet:

Data Set
========

.. currentmodule:: Data.DataSet

The DataSet object is the interface between the data files and the data treatment and visualisation. It holds one data reader per scan, all of one instrument and measured on the same sample, and reads the files it is given with the reader of that instrument::

    from MJOLNIR.Data.DataSet import DataSet
    from MJOLNIR.Data.InstrumentName import InstrumentName

    ds = DataSet(InstrumentName.CAMEA, files=['camea2024n001486.hdf', 'camea2024n001487.hdf'])

The files are calibrated and converted into reciprocal space when they are added; nothing is written to disk. The binning is changed with ``set_binning``. Values of all files are returned together: ``ds.gather('h')`` gives the unmasked values of one quantity (see ``ArrayType`` for the names), and iterating over the data set gives its readers.

Cuts return a ``CutResult``: the binned data as a pandas DataFrame (``result.data``), the bin edges (``result.bins``) and the arguments of the cut (``result.parameters``). Each ``plot_...`` method makes the cut, plots it and returns the axes, with the cut in ``ax.result``.

.. autosummary::
   :nosignatures:

    DataSet.DataSet
    DataSet.DataSet.set_binning
    DataSet.DataSet.gather
    DataSet.DataSet.mask
    DataSet.DataSet.cut_1d
    DataSet.DataSet.plot_cut_1d
    DataSet.DataSet.cut_constant_q
    DataSet.DataSet.plot_cut_constant_q
    DataSet.DataSet.cut_qe
    DataSet.DataSet.plot_cut_qe
    DataSet.DataSet.cut_qe_line
    DataSet.DataSet.plot_cut_qe_line
    DataSet.DataSet.cut_powder
    DataSet.DataSet.plot_cut_powder
    DataSet.DataSet.cut_q_plane
    DataSet.DataSet.plot_q_plane
    DataSet.DataSet.bin_3d
    DataSet.DataSet.view_3d
    DataSet.DataSet.generate_background_model
    DataSet.DataSet.absolute_normalize


.. automodule:: Data
   :members:



DataSet Object and Methods
--------------------------

Object holding the data readers of a set of scans and making cuts and plots of their data in reciprocal space.

.. _DataSet:

.. automodule:: DataSet
   :members:
