Quick 3D View of data
=====================

During an experiment a quick overview of the current data is often times needed. By the use of the *MJOLNIR3DView* script this is possible without using python directly. 

The script *MJOLNIR3DView* and it has the following help text::

    $ MJOLNIR3DView -h
    usage: MJOLNIR3DView [-h] [-r] [-b BINNING] [-d DQXDQYDE DQXDQYDE DQXDQYDE]
                         [-M VMAX] [-m VMIN]
                         [-i {CAMEA,BAMBUS,FlatCone,Marmot,MultiFLEXX}]
                         [DataFile ...]

    Conversion tool for quick visualization using the viewer3D.

    positional arguments:
      DataFile              Data file(s) to be used. If none provided file
                            dialogue will appear. Using string format, directory
                            and year is also possible. See documentation.

    options:
      -h, --help            show this help message and exit
      -r, --reuse           Set flag to reuse files from previous usage. Default
                            false.
      -b, --binning BINNING
                            Binning performed. Default the instrument's (8 for
                            CAMEA)
      -d, --dQxdQydE DQXDQYDE DQXDQYDE DQXDQYDE
                            Binning used to plot in 3D, Default [0.03,0.03,0.08]
      -M, --VMax VMAX       Maximal value for plotting, default max of data
      -m, --VMin VMIN       Minimal value for plotting, default min of data
      -i, --instrument {CAMEA,BAMBUS,FlatCone,Marmot,MultiFLEXX}
                            Instrument of the data files. Default 'CAMEA'

As input, one can either specify all of the data files wanted, use the number formatting explained below or simply leave out these; then a dialogue window will appear. If data files have previously been specified and one wants to reuse
the files, simply add the -r flag without any data files.
To specify the prismatic analyser binning, the -b flag is used, while -d is used to specify the bin sizes along the first scattering plane vector in 1/AA, along second in 1/AA, and along energy in meV.
If the colour scale is non-optimal, one can use the -m and -M flags to specify lower and upper limit. Files of other instruments than CAMEA are read with the -i flag.

The data are shown in the slice viewer of ``DataSet.view_3d``, in RLU units: constant-energy slices on reciprocal lattice axes, with a slider, the arrow keys and the scroll wheel to move through the slices and the keys 0, 1 and 2 to change the slicing axis. With cutting switched on (key c), lines drawn on a slice become 1D, Q-E or constant-Q cuts, which are plotted in new windows. If other properties or layouts are needed, take a look at the scripting tutorial for ``view_3d`` (AdvancedView3DTutorial).

The script needs PyQt5, for the file dialogue; it is not installed with MJOLNIR.

Number formatting
^^^^^^^^^^^^^^^^^

Specifying a complete data path for each individual file is at best annoying, thus one can use the shorter way of specifying the data files numbers, their folder, and the year they were taking. Later on also 
multiple instruments are supposed to be supported. The directory and year are believed to be self-explanatory, but the number string has the format as::

    numberString1 = '10-12,234,223,24-30'
    numberString2 = '10,11,12,13' 
    numberString3 = '10-13' 

That is, one can specify single files, ranges of files, or a combination separated with commas. In the above example *numberString2* and *numberString3* are equivalent.
