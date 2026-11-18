History of data files
=====================

Both during an experiment and especially after, one can lose the overview of which data files contain which data. With a 
command line tool that prints out the most important properties of data files, this can be overcome.

The script *MJOLNIRHistory* and it has the following help text::

    $ MJOLNIRHistory -h
    usage: MJOLNIRHistory [-h] [-s SAVE] [-r]
                          [-i {CAMEA,BAMBUS,FlatCone,Marmot,MultiFLEXX}]
                          [DataFile ...]

    History tool for displaying files and command for selected data files.

    positional arguments:
      DataFile              Data file(s) to be used. If none provided file
                            dialogue will appear. Using string format, directory
                            and year is also possible. See documentation.

    options:
      -h, --help            show this help message and exit
      -s, --save SAVE       Location to which the generated history will be saved.
      -r, --reuse           Set flag to reuse files from previous usage. Default
                            false.
      -i, --instrument {CAMEA,BAMBUS,FlatCone,Marmot,MultiFLEXX}
                            Instrument of the data files. Default 'CAMEA'


This script prints out one line for each data file selected containing the name of the file, its scan command and its title. Only these values are read from the files, so it is quick also for many files. Files of other instruments are read with the -i flag. With -s the lines are also saved to a text file (.txt is added if missing; an existing file is renamed with "_old" first). A file that cannot be read gives the line "name not correct format".

As an example, running this on the MnF2 data file camea2018n000500.hdf one gets the following output::

    $ MJOLNIRHistory camea2018n000500.hdf
    camea2018n000500.hdf: sc a3 50 da3 1 np 141 mn 50000	MnF2 MV=80 Ei=10 2t=-20 10 K

