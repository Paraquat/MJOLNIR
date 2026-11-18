Data Readers
============

A data reader reads one scan of one instrument and converts it into reciprocal space. Each instrument has its own reader, all derived from ``DataReader``; ``DataReaderFactory`` finds the reader of an instrument (``InstrumentName``) and reads files with it. A ``DataSet`` creates the readers itself when it is given file names::

    from pathlib import Path
    from MJOLNIR.Data.DataReaderFactory import DataReaderFactory
    from MJOLNIR.Data.InstrumentName import InstrumentName

    factory = DataReaderFactory()
    reader = factory.create(InstrumentName.CAMEA, Path('camea2024n001486.hdf'))
    values = factory.shallow_read(InstrumentName.CAMEA, [Path('camea2024n001486.hdf')],
                                  ['scanCommand', 'title'])

A reader holds the values as read (``counts``, ``A3``, ``A4``, ``Ei``, monitors, the sample, the sample environment and the metadata). ``set_instrument_calibration`` chooses the binning and ``transform`` converts the data: ``intensity``, ``monitor``, ``normalization``, ``energy``, ``qx``, ``qy``, ``h``, ``k`` and ``l``, one value per point (scan step, detector, energy bin). ``shallow_read`` reads a few values of many files without reading the data.

The Nexus readers (CAMEA, Flatcone, Marmot) read their files through a layout (``NexusLayout``): where each value is stored, the older paths also accepted, and its units and type. The NICOS readers (BAMBUS, MultiFLEXX) read NICOS data files.

.. contents::
   :local:
   :depth: 1


Base class
----------

.. automodule:: Data.DataReader
   :members:


Factory and instrument names
----------------------------

.. automodule:: Data.DataReaderFactory
   :members:

.. automodule:: Data.InstrumentName
   :members:
   :undoc-members:


Array names
-----------

``ArrayType`` names the arrays of a reader that a DataSet can gather and masks can use as coordinates.

.. automodule:: Data.ArrayType
   :members:
   :undoc-members:


Nexus readers
-------------

.. automodule:: Data.NexusDataReader
   :members:

.. automodule:: Data.CameaDataReader
   :members:

.. automodule:: Data.FlatconeDataReader
   :members:

.. automodule:: Data.MarmotDataReader
   :members:


Nexus layouts
-------------

.. automodule:: Data.NexusLayout
   :members:


NICOS readers
-------------

.. automodule:: Data.NicosDataReader
   :members:

.. automodule:: Data.BambusDataReader
   :members:

.. automodule:: Data.MultiFlexxDataReader
   :members:


Metadata and legacy fixes
-------------------------

.. automodule:: Data.Metadata
   :members:

.. automodule:: Data.LegacyFixes
   :members:
