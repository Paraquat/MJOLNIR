Data Writers and Export
=======================

Data files are not changed by MJOLNIR: a writer saves a copy of a file with new values. ``CameaDataWriter`` saves a CAMEA file with new instrument calibrations (normalisation tables), e.g. after ``DataSet.update_calibration``; the values it does not change are copied from the original file::

    from MJOLNIR.Data.CameaDataWriter import CameaDataWriter

    reader = ds[0] # A CameaDataReader of a DataSet
    CameaDataWriter(reader).save('camea2024n001486_calibrated.hdf')

Cut results are exported to ufit with ``DataSet.generate_ufit_dataset`` (``Export``), and the plots made by MJOLNIR to CSV files with ``ax.to_csv(fileName)``.

.. contents::
   :local:
   :depth: 1


Writers
-------

.. automodule:: Data.DataWriter
   :members:

.. automodule:: Data.NexusDataWriter
   :members:

.. automodule:: Data.CameaDataWriter
   :members:


Export
------

.. automodule:: Data.Export
   :members:
