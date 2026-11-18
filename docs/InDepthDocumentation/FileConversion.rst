.. _`Data file conversion`:

Data file conversion
====================

In general, it is expected that a CAMEA-like instrument is to be run during experiments in 2 different scan modes:

    - :math:`A3` scans

    - External parameter

However, in the initial phase of set-up other scans might be conducted. The data conversion thus 
does not require a specific scan but allows for all types. This does then require the user to choose visualizations correspondingly.

.. The most common operation is expected to be a rotate of the back-end into a suitable :math:`A4` and :math:`E_i` position to cover the interesting physics, and then a performance of an :math:`A3` scan. The rotation angle of this scan depends on the symmetry of the crystal in the given scattering plane as performing a 360 degrees scan with a 90 degrees symmetry does not provide additional information. After such a scan, would rotate :math:`A4` by half a wedge coverage angle (3.75 degrees) to cover the dark angles and then perform an identical :math:`A3` scan. This could be performed with different incoming energies to expand the covered area in the energy direction.

.. Having the raw data in the H5 format, converting the data files into :math:`S(\vec{q},\omega)` is rather straight forward. 

HdF file format
---------------

The raw data from the instrument is expected to be provided in an HdF 5 format with the following structure::

    cameasim2018n0000xx.h5
        └── entry
            ├── CAMEA
            │    ├── calib1
            │    │    ├── a4offset
            │    │    ├── amplitude
            │    │    ├── background
            │    │    ├── boundaries
            │    │    ├── final_energy
            │    │    └── width
            │    ├── calib3
            │    │    ├── a4offset
            │    │    ├── amplitude
            │    │    ├── background
            │    │    ├── boundaries
            │    │    ├── final_energy
            │    │    └── width
            │    ├── calib8
            │    │    ├── a4offset
            │    │    ├── amplitude
            │    │    ├── background
            │    │    ├── boundaries
            │    │    ├── final_energy
            │    │    └── width
            │    ├── detector
            │    │    ├── counts
            │    │    └── summed_counts
            │    ├── monochromator
            │    │    ├── d_spacing
            │    │    ├── energy
            │    │    ├── gm
            │    │    ├── gm_zero
            │    │    ├── horizontal_curvature
            │    │    ├── horizontal_curvature_zero
            │    │    ├── rotation_angle
            │    │    ├── rotation_angle_zero
            │    │    ├── summed_counts
            │    │    ├── tlm
            │    │    ├── tlm_zero
            │    │    ├── tum
            │    │    ├── tum_zero
            │    │    ├── type
            │    │    ├── vertical_curvature
            │    │    └── vertical_curvature_zero
            │    └── monochromator_slit
            │         ├── bottom
            │         ├── bottom_zero
            │         ├── left
            │         ├── left_zero
            │         ├── right
            │         ├── right_zero
            │         ├── top
            │         ├── top_zero
            │         ├── x_gab
            │         └── y_gab
            ├── comment
            ├── control
            │    ├── absolute_time
            │    ├── data
            │    ├── mode
            │    ├── preset
            │    └── time
            ├── data
            │    ├── counts
            │    ├── summed_counts
            │    └── (Scan parameter)
            ├── end_time
            ├── experimental_identifier
            ├── instrument
            ├── local_contact
            │    └── name
            ├── proposal_id
            ├── proposal_title
            ├── proposal_user
            │    └── name
            ├── proton_beam
            │    └── data
            ├── sample
            │    ├── azimuthal_angle
            │    ├── name
            │    ├── orientation_matrix
            │    ├── plane_normal
            │    ├── plane_vector_1
            │    ├── plane_vector_2
            │    ├── polar_angle
            │    ├── polar_angle_zero
            │    ├── rotation_angle
            │    ├── rotation_angle_zero
            │    ├── sgl
            │    ├── sgl_zero
            │    ├── sgu
            │    ├── sgu_zero
            │    ├── (sample environment parameters)
            │    └── unit_cell
            ├── scancommand
            ├── scanvars
            ├── start_time
            ├── title
            └── user
                 ├── address
                 ├── affiliation
                 ├── email
                 └── name

From this file, raw plotting and a conversion algorithm is possible.

Files written by NICOS differ in places: for example the counts are in ``detector/data`` (older files: ``detector/counts``), the monitor used for normalisation in ``monitor_2/data``, and the sample environment in ``sample/temperature`` or ``sample/T`` together with its logs (``temperature_log`` or ``T_log``). MJOLNIR reads both; ``CameaLayout`` in ``MJOLNIR/Data/NexusLayout.py`` lists every path it reads and the older paths it also accepts.



Conversion
----------

Converted files are not written: a data file is converted in memory when it is added to a DataSet (``DataReader.transform``). The counts are rebinned into the energy bins of the chosen binning (``DataSet.set_binning``; the binnings a file holds are those of its ``calib`` groups), and each point gets its normalisation and monitor and its position: :math:`Q_x`, :math:`Q_y`, H, K, L and energy transfer. Earlier versions of MJOLNIR saved the result as an NXsqom file (.nxs); such files are no longer read or written.

The converted data are used through the DataSet, e.g. ``ds.gather('h')``, and cuts and plots can be saved as CSV files (``ax.to_csv(fileName)`` on the plots made by MJOLNIR). A raw data file with a new normalisation table is saved with ``CameaDataWriter``, in the HdF format above, as a new file.
