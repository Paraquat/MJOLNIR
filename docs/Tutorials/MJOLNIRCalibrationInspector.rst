Calibration inspection
======================

Inspection of the current normalization table used to convert the raw data measured into actual scattering intensities is of great importance. It holds the information of energy and A4 value of all of the individual pixels and thus governs the convertion into reciprocal space. Furthermore, it contains the normalization needed to take care of detector sensitivity, space angle coverage, as well as analyser efficiency. Due to this vital role, it has been made possible to easily plot the contents of the calibration table.

The script *MJOLNIRCalibrationInspector* and it has the following help text::

    $ MJOLNIRCalibrationInspector -h
    usage: MJOLNIRCalibrationInspector [-h] [-s SAVE] [-p [PLOTLIST ...]]
                                       [-b BINNING]
                                       [-i {CAMEA,BAMBUS,FlatCone,Marmot,MultiFLEXX}]
                                       [DataFile]

    Inspection tool to visualize calibration tables in a data file.

    positional arguments:
      DataFile              Data file from which calibration table is to be
                            plotted. If none provided file dialogue will appear.

    options:
      -h, --help            show this help message and exit
      -s, --save SAVE       Location to which the generated file will be saved: a
                            folder, or a .png file when one plot is made.
      -p, --plot [PLOTLIST ...]
                            List of wanted plots to be generated. Should be
                            "A4","Normalization","Ef","EfOverview". Default all of
                            them.
      -b, --binning BINNING
                            Binning to be inspected. Default the instrument's (8
                            for CAMEA)
      -i, --instrument {CAMEA,BAMBUS,FlatCone,Marmot,MultiFLEXX}
                            Instrument of the data files. Default 'CAMEA'


First of all, if the program is run without any arguments, a file dialog will appear asking for a file to be plotted. From this file, the default plots will be created and the file directory is saved in an external settings file such that next time the program is run the file dialog will open on this location. The script needs PyQt5, for the file dialogue; it is not installed with MJOLNIR.

As seen in the description, the user can choose which plots are to be generated and whether to save the results or simply inspect them visually. As a default the script uses the instrument's default binning (8 pixels for CAMEA) and creates all four plots.
This can be changed by invoking the *-b binning* or *-p PlotType* flags for binnning and plotting respectively. The binning chosen should be one of those provided in the file (usually 1, 3, or 8; otherwise the script stops and lists them), and the plotting possibilities are "A4","Normalization","Ef", and "EfOverview" (in any case). Files of other instruments are read with the -i flag.

Providing a saving location with the use of the *-s savelocation* or *--save savelocation* makes the script save the figures in the corresponding path. It should be a directory, which is created if needed; when only one plot is made it can also be a .png file path.
The figures are saved with selfexplanartory titles, followed by the binning, in a png format such that they can be opened on all OS's without problems.

An example of the four different figures produces is the following, when choosing binning 1, 3, and 8:

Binning 1
^^^^^^^^^
|bin1EI| |bin1EO|
|bin1C| |bin1N|

.. |bin1EI| image:: Figures/Final_Energy_Individual_1.png
   :width: 49%

.. |bin1EO| image:: Figures/Final_Energy_Overview_1.png
   :width: 49%

.. |bin1C| image:: Figures/Instrument_calibration_1.png
   :width: 49%

.. |bin1N| image:: Figures/Normalization_1.png
   :width: 49%

Binning 3
^^^^^^^^^
|bin3EI| |bin3EO|
|bin3C| |bin3N|

.. |bin3EI| image:: Figures/Final_Energy_Individual_3.png
   :width: 49%

.. |bin3EO| image:: Figures/Final_Energy_Overview_3.png
   :width: 49%

.. |bin3C| image:: Figures/Instrument_calibration_3.png
   :width: 49%

.. |bin3N| image:: Figures/Normalization_3.png
   :width: 49%

Binning 8
^^^^^^^^^
|bin8EI| |bin8EO|
|bin8C| |bin8N|

.. |bin8EI| image:: Figures/Final_Energy_Individual_8.png
   :width: 49%

.. |bin8EO| image:: Figures/Final_Energy_Overview_8.png
   :width: 49%

.. |bin8C| image:: Figures/Instrument_calibration_8.png
   :width: 49%

.. |bin8N| image:: Figures/Normalization_8.png
   :width: 49%
