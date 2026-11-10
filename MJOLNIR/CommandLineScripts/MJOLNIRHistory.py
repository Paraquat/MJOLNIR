#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:46:53 2018

@author: Jakob Lass

History generator tool for looking through data files and create history.
"""


import argparse
import os
from pathlib import Path

settingsName = 'HistoryDir'

from MJOLNIR.CommandLineScripts import _tools
from MJOLNIR.Data.DataReaderFactory import DataReaderFactory


parser = argparse.ArgumentParser(description="History tool for displaying files and command for selected data files.")
parser.add_argument("DataFile", nargs ='*', default=argparse.SUPPRESS, type=str,help="Data file(s) to be used. If none provided file dialogue will appear. Using string format, directory and year is also possible. See documentation.")
parser.add_argument("-s", "--save", type=str, default= '',help="Location to which the generated history will be saved.")
parser.add_argument("-r", "--reuse", action='store_true', help='Set flag to reuse files from previous usage. Default false.')
_tools.addInstrumentArgument(parser)


def asText(value):
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def historyLine(file, instrument, factory=None):
    """'name: scan command<tab>title' of a data file, or 'name not correct format'"""
    if factory is None:
        factory = DataReaderFactory()
    try:
        values = factory.shallow_read(instrument, [Path(file)], ['name', 'scanCommand', 'title'])[0]
    except Exception:
        return os.path.basename(file)+' not correct format'
    return values['name']+': '+asText(values['scanCommand'])+'\t'+asText(values['title'])


def main(argv=None):
    args = parser.parse_args(argv)
    files = _tools.extractDataFiles(args,settingsName)
    _tools.updateSetting(settingsName,files)
    instrument = _tools.instrumentName(args)

    factory = DataReaderFactory()
    lines = []
    for file in files:
        line = historyLine(file, instrument, factory)
        print(line)
        lines.append(line+'\n')

    saveFile = args.save.strip()
    if saveFile != '':
        if saveFile.split('.')[-1]!='txt':
            saveFile+='.txt'
        if os.path.isfile(saveFile):
            oldName = '_old.'.join([x for x in saveFile.split('.')])
            print('File already exists. Renaming old file to {}'.format(oldName))
            os.rename(saveFile,oldName)
        with open(saveFile,'w') as f:
            f.write(''.join(lines))


if __name__ == '__main__':
    main()
