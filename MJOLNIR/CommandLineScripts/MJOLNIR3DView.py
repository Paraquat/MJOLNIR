#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:24:53 2019

@author: Jakob Lass

Tool for quick viewing of data with viewer3D
"""


import argparse
import matplotlib.pyplot as plt

from MJOLNIR.CommandLineScripts import _tools
settingsName = 'ConvertionDir'


from MJOLNIR.Data.DataSet import DataSet


parser = argparse.ArgumentParser(description="Conversion tool for quick visualization using the viewer3D.")
parser.add_argument("DataFile", nargs ='*', default=argparse.SUPPRESS, type=str,help="Data file(s) to be used. If none provided file dialogue will appear. Using string format, directory and year is also possible. See documentation.")
parser.add_argument("-r", "--reuse", action='store_true', help='Set flag to reuse files from previous usage. Default false.')
parser.add_argument("-b", "--binning", type=int, default=None,help="Binning performed. Default the instrument's (8 for CAMEA)")
parser.add_argument("-d", "--dQxdQydE", nargs=3, type=float, default=[0.03,0.03,0.08], help="Binning used to plot in 3D, Default [0.03,0.03,0.08]")
parser.add_argument("-M", "--VMax", type=float, default=None, help='Maximal value for plotting, default max of data')
parser.add_argument("-m", "--VMin", type=float, default=None, help='Minimal value for plotting, default min of data')
_tools.addInstrumentArgument(parser)


def view(args):
    """The data set of the files in args and its View3D viewer"""
    files = _tools.extractDataFiles(args,settingsName)
    _tools.updateSetting(settingsName,files)

    dataSet = DataSet(instrument=_tools.instrumentName(args), files=list(files))
    if args.binning is not None:
        dataSet.set_binning(args.binning)
    dQx, dQy, dE = args.dQxdQydE
    V = dataSet.view_3d(dQx, dQy, dE, rlu=True, vmin=args.VMin, vmax=args.VMax)
    return dataSet, V


def main(argv=None):
    args = parser.parse_args(argv)
    dataSet, V = view(args)
    plt.show()


if __name__ == '__main__':
    main()
