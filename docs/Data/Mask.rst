Masks
=====

A mask excludes points of the data from cuts and plots: ``True`` excludes a point. ``DataSet.mask`` takes one boolean array per data file, or a mask object from this module. Mask objects describe regions (circles, boxes, ... in any coordinates of the data, e.g. ``h``, ``k``, ``l``, ``qx``, ``energy`` or ``A3``) and are combined with ``+`` (points in either mask), ``*`` (points in both) and negated with a unary ``-``; the DataSet evaluates them again when the binning changes or files are added. ``CurratAxeMask`` masks the spurions of Bragg peaks. See the Masking tutorial for examples.

Every data file also has a default mask of the points its instrument cannot use (e.g. broken detectors), which applies unless ``DataSet.use_default_mask`` is ``False``.

.. automodule:: Data.Mask
   :members:
