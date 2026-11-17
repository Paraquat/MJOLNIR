"""Corrections for data files written with a known fault, kept apart from the readers.

A legacy fix is tied to where a file came from (a year, an old writer), so the set of files
it applies to no longer grows. Each reader lists its fixes in `legacy_fixes`; the reader runs
them once it has read a file (DataReader.apply_legacy_fixes), warns for each fix applied and
records it in metadata['legacy_fixes']. shallow_read still gives the values as stored.

Handling that current files can need (e.g. a scan still running) belongs in the reader, not
here."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from MJOLNIR.Data.DataReader import DataReader

class LegacyFix(ABC):
    """A correction for files written with a known fault"""

    description: str # What the fix does, recorded in the reader's metadata

    @abstractmethod
    def applies(self, reader: DataReader) -> bool:
        """Whether the reader's file has the fault"""

    @abstractmethod
    def apply(self, reader: DataReader) -> None:
        """Correct the reader's values"""

class CameaMonitor2018(LegacyFix):
    """CAMEA files of 2018 counted on monitor stored a wrong monitor: the old DataFile
    replaced it with the preset ("TODO: Make Mark save the correct monitor"). Applied, as
    there, when the file is from 2018, has more than one scan step, and its first monitor
    value differs from the preset. Unlike the old code, not in time mode: its check of the
    mode compared bytes with a string, so it also replaced the monitor of time-mode files
    with a preset in seconds. A file without a mode is taken as counted on monitor, as there"""

    description = '2018 monitor replaced by the preset'

    def applies(self, reader: DataReader) -> bool: # A CameaDataReader
        metadata = reader._metadata
        preset = metadata.get('preset')
        if not str(metadata.get('startTime', '')).startswith('2018') or preset is None:
            return False
        if str(metadata.get('mode', '')).lower() in ('t', 'time'):
            return False
        monitor = np.asarray(getattr(reader, reader.scan_monitor_key()), dtype=float).ravel()
        return monitor.size > 1 and monitor[0] != preset

    def apply(self, reader: DataReader) -> None:
        key = reader.scan_monitor_key()
        monitor = np.asarray(getattr(reader, key), dtype=float)
        setattr(reader, key, np.full_like(monitor, float(reader._metadata['preset'])))
