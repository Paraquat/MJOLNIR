Plotting
========

``Plotting`` draws cut results: each function takes a ``CutResult`` and axes (new axes if none are given) and returns the axes, with the result in ``ax.result`` and ``ax.to_csv(fileName)`` to save it. The ``DataSet`` methods ``plot_cut_1d``, ``plot_cut_qe``, ``plot_q_plane``, ... make the cut and call these functions. A click on a plot prints a read-out of the bin under the mouse (see `Interactivity <../InDepthDocumentation/Interactivity.html>`_).

``RLUAxes`` creates axes in reciprocal lattice units: the data are drawn in Qx, Qy and the axes show H, K, L (``DataSet.create_q_axis``, ``DataSet.create_qe_axes``). ``DraggableShapes`` turns rectangles and circles drawn on plots into the arguments of the DataSet cuts, for programs drawing cuts on plots such as MJOLNIR-GUI. The 3D viewer is described in the `Graphical interface <Gui.html>`_ page.

.. contents::
   :local:
   :depth: 1


Plotting functions
------------------

.. automodule:: Data.Plotting
   :members:


Reciprocal lattice axes
-----------------------

.. automodule:: Data.RLUAxes
   :members:


Cuts from drawn shapes
----------------------

.. automodule:: Data.DraggableShapes
   :members:
