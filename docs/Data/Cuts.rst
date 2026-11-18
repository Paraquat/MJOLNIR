Cuts
====

The cuts are functions of arrays of data points (``Cuts``): they bin the points and return a ``CutResult``, without knowing about data files or data sets. The ``DataSet`` methods (``cut_1d``, ``cut_qe``, ...) collect the unmasked points of their files, convert positions given in H, K, L, and call these functions.

A ``CutResult`` holds the binned data as a pandas DataFrame (``result.data``, one row per bin, with the columns listed below), the bin edges (``result.bins``) and the arguments of the cut (``result.parameters``), so that a cut can be made again with changed arguments, e.g. ``ds.cut_1d(**{**result.parameters, 'width': 0.2})``.

``Coordinates`` converts between H, K, L and the instrument frame (Qx, Qy) and rotates data files into a common frame. ``AxisView`` presents the arrays of several data files as one array, as returned by ``DataSet.get_view`` and ``DataSet.gather``.

.. contents::
   :local:
   :depth: 1


Cut functions
-------------

.. automodule:: Data.Cuts
   :members:


Cut results
-----------

.. automodule:: Data.CutResult
   :members:


Coordinates
-----------

.. automodule:: Data.Coordinates
   :members:


Arrays of several files
-----------------------

.. automodule:: Data.AxisView
   :members:
