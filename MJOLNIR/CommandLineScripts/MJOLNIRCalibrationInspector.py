#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:46:53 2018

@author: Jakob Lass

Inspection tool to open the normalization tables saved in a given data file.
"""


import argparse
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

from MJOLNIR.CommandLineScripts import _tools
settingsName = 'CalibrationInspectorDir'

from MJOLNIR.Data import Plotting
from MJOLNIR.Data.DataReaderFactory import DataReaderFactory


PlotType = {
        "A4": Plotting.plot_calibration_a4,
        "Normalization": Plotting.plot_calibration_normalization,
        "Ef": Plotting.plot_calibration_ef,
        "EfOverview": Plotting.plot_calibration_ef_overview,
    }


def switch(argument):
    for plotType in PlotType:
        if plotType.upper() == argument.upper():
            return plotType
    warnings.warn('Plot type "{}" not understood and will be skipped.'.format(argument))


parser = argparse.ArgumentParser(description="Inspection tool to visualize calibration tables in a data file.")
parser.add_argument("DataFile", nargs ='?', default=argparse.SUPPRESS, type=str,help="Data file from which calibration table is to be plotted. If none provided file dialogue will appear.")
parser.add_argument("-s", "--save", type=str, default= '',help="Location to which the generated file will be saved: a folder, or a .png file when one plot is made.")
parser.add_argument('-p', "--plot", nargs = '*', dest = 'plotList', help = 'List of wanted plots to be generated. Should be "{}". Default all of them.'.format('","'.join(PlotType)), default = list(PlotType))
parser.add_argument("-b", "--binning", type=int, default=None,help="Binning to be inspected. Default the instrument's (8 for CAMEA)")
_tools.addInstrumentArgument(parser)


def inspect(args):
    """Plot the calibration tables of the file in args; returns the axes by plot type"""
    file = _tools.extractDataFiles(args,settingsName,oneFile=True)
    if isinstance(file, (list, tuple)):
        file = file[0]
    _tools.updateSetting(settingsName,os.path.split(file))

    reader = DataReaderFactory().create(_tools.instrumentName(args), Path(file))
    binning = reader.default_binning if args.binning is None else args.binning
    if binning not in reader.possible_binnings:
        raise ValueError('Binning {} not in the possible binnings {} of "{}"'.format(binning, reader.possible_binnings, file))
    calibration = reader.get_instrument_calibration(binning)

    plots = []
    for arg in args.plotList:
        plotType = switch(arg)
        if plotType is not None and plotType not in plots:
            plots.append(plotType)
    axes = {plotType: PlotType[plotType](calibration, reader.n_detectors) for plotType in plots}

    save = args.save.strip()
    if save != '' and len(axes) > 0:
        if len(axes) == 1 and save.lower().endswith('.png'):
            locations = {plotType: save for plotType in axes}
        else:
            os.makedirs(save, exist_ok=True)
            locations = {plotType: os.path.join(save, ax.get_title().replace(' ','_')+'_'+str(binning)+'.png')
                         for plotType, ax in axes.items()}
        for plotType, ax in axes.items():
            print('Save figure as '+locations[plotType])
            ax.get_figure().savefig(locations[plotType])
    return axes


def main(argv=None):
    args = parser.parse_args(argv)
    inspect(args)
    plt.show()


if __name__ == '__main__':
    main()
