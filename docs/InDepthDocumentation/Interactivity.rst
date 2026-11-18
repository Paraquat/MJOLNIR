.. :Interactivity:

Interactivity
^^^^^^^^^^^^^

Working with the 3D data from CAMEA can be a challenge. Thus, a set of tools have been created to inspect the data directly on the plots: a read-out of the data under the mouse, cuts drawn on the slices of the 3D viewer, and the resolution ellipsoid at a point of interest.

.. table:: Overview of the interaction possibilities in MJOLNIR
    :align: center

    +-----------------------+-------------------------------------------+------------------------+
    | Interaction           | Where                                     | How                    |
    +=======================+===========================================+========================+
    | Read-out              | All cut plots and the 3D viewer           | Left click             |
    +-----------------------+-------------------------------------------+------------------------+
    | Browse slices         | 3D viewer (``view_3d``)                   | Slider, arrows, scroll |
    +-----------------------+-------------------------------------------+------------------------+
    | Cutting               | 3D viewer (``view_3d``)                   | c, then click and drag |
    +-----------------------+-------------------------------------------+------------------------+
    | Resolution            | Any plot, from a script                   | ``resolution_ellipse`` |
    +-----------------------+-------------------------------------------+------------------------+

The interactive modes of earlier versions on the Q plane and Q-E plots (resolution mode 'r', cutting mode 'c' with the manage mode 'm', cycling of cut types 'n', snapping, and 2D cuts with 'Ctrl + c') are not available in this version. Cuts are drawn in the 3D viewer, described below.


Read-out
--------

Clicking with the left mouse button on a plot made by MJOLNIR (``plot_cut_1d``, ``plot_cut_constant_q``, ``plot_cut_qe``, ``plot_cut_qe_line``, ``plot_cut_powder``, ``plot_q_plane`` and the 3D viewer) prints the position under the mouse and the values of the bin there: the normalised intensity, the counts, the normalisation, the monitor and the number of data points. The plotting methods take an ``on_click`` argument, called with the read-out text (default ``print``); ``on_click=None`` switches the read-out off. Clicks while zooming or panning are ignored.


Viewer 3D and cutting
---------------------

``DataSet.view_3d(dQx, dQy, dE)`` bins the data in voxels and shows one slice at a time: constant energy (axis 2, on reciprocal lattice axes in RLU), constant :math:`Q_x` (axis 0) or constant :math:`Q_y` (axis 1). For details see the AdvancedView3DTutorial.

.. table:: Interaction possibilities in the 3D viewer
    :align: center

    +-------------------------+------------------------------+
    | Method                  | Button                       |
    +=========================+==============================+
    | Next/previous slice     | Up/right, down/left, scroll  |
    +-------------------------+------------------------------+
    | Slicing axis            | 0, 1, 2                      |
    +-------------------------+------------------------------+
    | Cutting on/off          | c                            |
    +-------------------------+------------------------------+
    | Draw cut                | Left click and drag          |
    +-------------------------+------------------------------+
    | Read-out (cutting off)  | Left click                   |
    +-------------------------+------------------------------+

With cutting switched on, a line drawn on a slice becomes a cut. On a constant-energy slice it is a 1D cut (``cut_1d``) from the start to the end of the line, over the energy range of the slice. On a Q-E slice a line is a Q-E cut (``cut_qe``) over the energies it spans, and a click without dragging (or a vertical line) an energy cut at constant Q (``cut_constant_q``). The width of the cuts and their bin size along Q are set with the ``cut_width`` and ``cut_min_pixel`` arguments of ``view_3d``. By default each cut is plotted in a new figure with the data set's own method; a function given as ``on_cut`` receives instead the kind of cut ('1d', 'qe' or 'constant_q') and its arguments, e.g. to collect the cuts or make them differently.


Resolution
----------

The resolution ellipsoid can be calculated at any point of the data. It utilizes an updated version of the method presented in
[eck14] G. Eckold and O. Sobolev, NIM A 752, pp. 54-64 (2014) and has been implemented `Tobias Weber <tweber@ill.fr>`_ in the software `TARKIN <https://doi.org/10.5281/zenodo.4117437>`_.

The ellipsoid is inherently 4D (dimensions are Qx, Qy, Qz, and energy), but the plots upon which the resolution is shown is 2D. By integration over the two coordinates not
used in the figure, one can find the resolution matrix in the relevant plane. As this matrix (in this approximation) only dependent on the instrument, the sample rotation is to be taken into
account as well as the possible rescaling of the plotting axis.

``DataSet.resolution_matrix(position, Ei, Ef)`` gives the 4 by 4 matrix at a position (H, K, L, or Qx, Qy with ``rlu=False``), and ``DataSet.resolution_ellipse`` its projection onto the plane of two directions in (Qx, Qy, Qz, energy): the matrix, the axes of the ellipse (eigenvectors as columns) and the widths (standard deviations) along them. With ``lattice_axes=True`` the axes are rotated for plots on reciprocal lattice axes. They are implemented for CAMEA only. Below the ellipse in the Qx-Qy plane is drawn on a constant-energy plot:

.. code-block:: python

    import numpy as np
    from matplotlib.patches import Ellipse

    ax = ds.plot_q_plane(energy_min=0.0, energy_max=0.2, rlu=False)

    position = [1.0, 0.5] # Qx, Qy [1/AA]
    M, axes, sigma = ds.resolution_ellipse(position, [1, 0, 0, 0], [0, 1, 0, 0],
                                           Ei=4.0, Ef=3.9, rlu=False)
    axes, sigma = np.real(axes), np.real(sigma)
    angle = np.rad2deg(np.arctan2(axes[1, 0], axes[0, 0]))
    fwhm = 2*np.sqrt(2*np.log(2))*sigma # Full widths at half maximum
    ax.add_patch(Ellipse(position, fwhm[0], fwhm[1], angle=angle, fill=False, color='r'))

If two different incoming energies overlap at the point, the resolution is to be calculated for each of them.
