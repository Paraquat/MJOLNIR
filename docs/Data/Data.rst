Data Module
===========

The Data module reads data files and treats their data. A data reader per instrument (``CameaDataReader``, ``BAMBUSDataReader``, ``FlatconeDataReader``, ``MarmotDataReader``, ``MultiFLEXXDataReader``, chosen by ``DataReaderFactory``) reads one scan and converts it into reciprocal space; a DataSet holds the readers of several scans and makes the cuts and plots. Cuts are computed in ``Cuts`` and returned as a ``CutResult``, and plotted by ``Plotting``.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   DataSet
   Sample
   Gui


