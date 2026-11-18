FAQ
===

A collection of questions and answers meant to guide the user(s) of the MJOLNIR package through some of the most common pitfalls.


My RLU-axis has its tick marks positions stupidly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This 'feature' of the RLU axis is unfortunately linked to the current stage of the development of the `GridHelperCurveLinear <https://matplotlib.org/api/_as_gen/mpl_toolkits.axisartist.grid_helper_curvelinear.GridHelperCurveLinear.html#mpl_toolkits.axisartist.grid_helper_curvelinear.GridHelperCurveLinear>`_.
Because of the current lack of customization for this in-progress work some attributes cannot be changed. However, when generating an new RLUaxis 
one can specify the number of tick marks on the x and y axis. This number stays constant  throughout the lifetime of that axis, but a new can be created with a 
different number if needed. Further, specifying only the number of y-axis tick marks the program calculates a suitable number along the x-axis.


I do not want to make use of the features in MJOLNIR but only convert and extract my data. How do I do this?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is definitely possible to convert and extract the data from MJOLNIR without many troubles. The data files are converted when they are added to a DataSet, and ``gather`` returns the unmasked values of one quantity of all files. Below is a code example:

.. code-block:: python
   :linenos:

    import numpy as np
    from MJOLNIR.Data.DataSet import DataSet
    from MJOLNIR.Data.InstrumentName import InstrumentName

    fileName = ['/Path/To/Data/camea2018n000136.hdf','/Path/To/Data/camea2018n000137.hdf']
    ds = DataSet(InstrumentName.CAMEA, files=fileName)
    ds.set_binning(3)

    values = {name: np.asarray(ds.gather(name))
              for name in ['h', 'k', 'l', 'energy', 'intensity', 'monitor', 'normalization']}
    Int = values['intensity']/(values['monitor']*values['normalization']) # Normalised intensity

    data = np.array([values['h'], values['k'], values['l'], values['energy'], Int])

    saveFile = '/Path/To/Data/Data.csv'
    np.savetxt(saveFile, data.T, delimiter=',', header='H,K,L,Energy,Int')

However, a small warning. The data file created from this can be of several hundred MB.



My data has annoying lines for constant energy that looks like background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When measuring data on CAMEA and utilizing the prismatic concept one can split the nominal eight final energies into a maximum of 64 energies. This is performed in the conversion and is encoded in the normalization
files saved in each data file. Due to difference in sensitivity of these prismatic pixels any uniform background on the detector tube will be inflated at the points of low sensitivity resulting in lines of 
higher scattering intensity for constant energies. These then show up as lines, both in 2D plots but more importantly especially in energy cuts. Currently. there are two solutions to be employed, one for during experiments 
and one for post experiment data treatment if first possibility is not done:

- When performing a measurement, interlace all A3/sample rotation scans with additional scans with an Ei slightly changed.

- Perform a powder background subtraction as explained in the `Background Subtraction Tutorial <Tutorials/Advanced/backgroundSubtraction.html>`_.

The amount of change needed in :math:`E_i` to counter the stripes can be difficult to figure out and depends on the question at hand. However, as a standard approach 
one has to shift with half the mean difference in final energy. This ensures that the parts of the detectors with low sensitivity will be covered by parts of the detector tube 
with better sensitivity. Taking this half average, one finds the value: 0.127 meV. However, if one is applying a high resolution measurement where only a subset
of the final energies are useful (on the energy-loss side) and one would move :math:`E_i` by half the average of the final energies in use. Most often one uses
the (approximate) 3.6, 3.4, and 3.2 meV final energies where a shift of ~0.094 meV. Lastly, one can also add multiple shifts in :math:`E_i` whith sizes of 0.064, 0.127, and 0.191 meV, 
corresponding to 0.5, 1.0, and 1.5 times the energy shift.




.. table:: Mean Final Energies (`Commissioning Report <https://aip.scitation.org/doi/10.1063/5.0128226>`_):
   :widths: auto
   :align: center

   +-----------+------------------+
   | Ef [meV]  | Difference [meV] |
   +===========+==================+
   |  3.200    |  N/A             |
   +-----------+------------------+
   |  3.382    |  0.182           |
   +-----------+------------------+
   |  3.574    |  0.192           |
   +-----------+------------------+
   |  3.787    |  0.213           |
   +-----------+------------------+
   |  4.035    |  0.248           |
   +-----------+------------------+
   |  4.312    |  0.277           |
   +-----------+------------------+
   |  4.631    |  0.319           |
   +-----------+------------------+
   |  4.987    |  0.356           |
   +-----------+------------------+



I aligned my crystal slightly off when setting up the experiment. Can I correct for an offset in A3?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is definitely possible through the code below, where 10 degrees is added to the offset. The files are read first, the offset changed, and the readers then given to the DataSet, which converts them with the new offset.

.. code-block:: python

    from pathlib import Path
    from MJOLNIR import _tools
    from MJOLNIR.Data.DataReaderFactory import DataReaderFactory
    from MJOLNIR.Data.DataSet import DataSet
    from MJOLNIR.Data.InstrumentName import InstrumentName

    files = '276-279'

    # Generate file locations as normal
    dataFiles = _tools.fileListGenerator(files,'/path/to/data/')

    factory = DataReaderFactory()
    readers = [factory.create(InstrumentName.CAMEA, Path(file)) for file in dataFiles]
    for reader in readers: # Loop through all files
        reader.A3Offset = reader.A3Offset + 10

    ds = DataSet(InstrumentName.CAMEA, files=readers)
    # Continue as normal

Because my crystal has a 120 degrees symmetry, we only scanned 120 degrees in our scans. How do I symmetrize it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When performing the Q scans most often it is not needed at all to scan 360 degrees as the scattering planes has a rotation symmetry of e.g. 90, 120, or the like. There is no method of the DataSet object to do this but
one can, in a script, load in the measured data twice where the second set of data files is rotated by the symmetry. For the example in this question, I will assume that one has measured the files '276-279' of 120 degrees and the system has 120 degrees symmetry, thus datafiles need to replicated twice and ritated 120 and 240 degrees:

.. code-block:: python

    files = '276-279'

    # Generate file locations as normal
    dataFiles = _tools.fileListGenerator(files,'/path/to/data/')
    numFiles = len(dataFiles) # The number of data files actually measured
    dataFiles = dataFiles+dataFiles+dataFiles # Repeat the file list to contain 3 copies

    # Read the 12 data files (imports as in the question above)
    factory = DataReaderFactory()
    readers = [factory.create(InstrumentName.CAMEA, Path(file)) for file in dataFiles]

    for i,reader in enumerate(readers): # Loop through all files and keep a counter
        # Files 4-7 get 120 degrees added to A3, files 8-11 240 degrees
        reader.A3Offset = reader.A3Offset + 120.0*(i//numFiles)

    ds = DataSet(InstrumentName.CAMEA, files=readers)

One then has a DataSet object with 12 data files covering the full scattering plane, the so-called donut plot. If a crystal has a symmetry smaller than what is measured (e.g. 110 degrees are measured but symmetry is 90 degrees), there is no problem with this procedure. It merely results in double data coverage at the positions in Q where there is an overlap. 


When I save my nice figures as .eps white lines appear around all pixels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This has to do with how Matplotlib shows figures in contrast to how the eps engine saves the file. To remove the lines add the key word argument 

.. code-block:: python

    linewidth=0,rasterized=True

to the plotting function. This then forces the eps back-end to draw the surrounding edges as the face colour instead of white/transparent.


I cannot change the colour scale when running MJOLNIR through the terminal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to be able to interact with the figures generated by MJOLNIR when using the terminal make sure to use the interactive flag of matplotlib. That is

.. code-block:: python

    import matplotlib.pyplot as plot
    plt.ion() # Set interactive matplotlib windows

This will make the generation of plots and the plt.show() non-blocking and thus allow the change of axes and other aspects of the plots.

.. _MaskFAQ:

I want to created a mask for my data to exclude specific points in Q
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When dealing with multiple data files at once in a DataSet object, one needs to keep in mind that MJOLNIR does not expect these to be of equal size. That is, there is no requirement for the number of step points to be equal. This then results in the mask of a DataSet being a list of np arrays, one per data file, each with the same shape as that file's data. Most easily, this is taken care of by looping through the data readers in the DataSet as:

.. code-block:: python

    ds = DataSet(InstrumentName.CAMEA, files=dataFiles)

    mask = []
    # Loop through all data files in the DataSet
    for reader in ds:
        mask.append(reader.h>0.0)

    ds.mask = mask

The above code ensures that all points with a value of H larger than 0.0 are masked out. Usually multiple conditions are required for the mask to be correctly created. As an example below is code that creates masks removing all data points within a radius of 0.1 1/A from the provided QPoints.


.. code-block:: python

    QPoints = [[1,1,0],[-1,1,0],[1,0,0],[0,1,0]]
    radius = 0.1
    mask = []

    for reader in ds:
    # calculate position in qx,qy for QPoints (may differ from file to file)
        localMask = []

        for h,k,l in QPoints:
            qx,qy = reader.sample.calculateHKLToQxQy(h,k,l)
            m = np.sqrt((reader.qx-qx)**2+(reader.qy-qy)**2)<radius
            localMask.append(m)
        trueMask = localMask[0]
        for m in localMask[1:]:
            trueMask = np.logical_or(trueMask,m)
        
        mask.append(trueMask)
    

or in one line

.. code-block:: python

    QPoints = [[1,1,0],[-1,1,0],[1,0,0],[0,1,0]]
    radius = 0.1
    from functools import reduce

    mask = [reduce(np.logical_or,[np.sqrt((reader.qx-qx)**2+(reader.qy-qy)**2)<radius for qx,qy in [reader.sample.calculateHKLToQxQy(*HKL) for HKL in QPoints]]) for reader in ds]

    ds.mask = mask

Masks can also be built from the mask objects of ``MJOLNIR.Data.Mask`` (e.g. circles and boxes in H, K, L or Qx, Qy), which the DataSet evaluates again when the binning changes or files are added; see the Masking tutorial.



After I have performed a cut, the data I receive as a data type of object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, when working with pandas DataFrames, the data type of the different columns might go from e.g. int64 to object. In order to change this
when wanting to perform numeric calculations, on can format the DataFrame to be numeric. A cut returns a CutResult whose ``data`` is the DataFrame. That is:

.. code-block:: python

    ...
    >>> result = ds.cut_1d(...)
    >>> result.data['intensity'].dtypes
    dtype('O')

    >>> intensity = pd.to_numeric(result.data['intensity'])
    >>> intensity.dtypes
    dtype('int64')


Is it possible to calculate the resolution of the instrument at a given position in reciprocal space?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Indeed it is, for CAMEA: ``DataSet.resolution_matrix`` gives the resolution matrix at a position and ``DataSet.resolution_ellipse`` its ellipse in a plane. How to overplot the resolution ellipse directly on your data is explained in `Interactivity <InDepthDocumentation/Interactivity.html>`_.


I want to make a cut directly on the 2D data that I have plotted. How can I do this?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cuts can be drawn on the slices of the 3D viewer (``DataSet.view_3d``): switch cutting on with 'c' and drag a line, as explained in `Interactivity <InDepthDocumentation/Interactivity.html>`_. On the plots of single cuts (e.g. ``plot_q_plane`` or ``plot_cut_qe``) cuts cannot be drawn in this version; there, a click prints the values under the mouse, whose position can be used for the next cut in a script.
