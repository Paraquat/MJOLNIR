Background, Normalisation and Resolution
========================================

``BackgroundModel`` holds the background models of a DataSet: a powder background (``DataSet.generate_background_model``), averaged over the length of Q and energy from the points not masked as signal, and the AMBER model (``DataSet.generate_amber_background``, which needs the AMBER-ds4ms package), which separates signal from a background depending on the length of Q and energy in the binned data. Cuts and plots subtract the model with ``background_subtraction=True``.

``AbsoluteNormalization`` scales intensities to absolute units against the vanadium normalisation (``DataSet.absolute_normalize``). ``Resolution`` calculates the resolution of CAMEA (``DataSet.resolution_matrix``, ``DataSet.resolution_ellipse``; see `Interactivity <../InDepthDocumentation/Interactivity.html>`_).

.. contents::
   :local:
   :depth: 1


Background models
-----------------

.. automodule:: Data.BackgroundModel
   :members:


Absolute normalisation
----------------------

.. automodule:: Data.AbsoluteNormalization
   :members:


Resolution
----------

.. automodule:: Data.Resolution
   :members:
