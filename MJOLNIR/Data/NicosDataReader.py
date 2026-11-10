import re
import numpy as np
import numpy.typing as npt
from pathlib import Path
from abc import abstractmethod

from MJOLNIR.Data.DataReader import DataReader
from MJOLNIR.Data.Sample import Sample

class NICOSDataReader(DataReader):
    """Parse a NICOS ASCII output file"""

    # NICOS names files by scan number, with or without .dat
    file_extensions = ('dat', '')

    # NICOS calibration tables store A4 with the opposite sign
    calibration_A4_sign = -1

    # Header keys for metadata fields
    nicos_meta_keys = {
        'title': 'Exp_title',
        'comment': 'Exp_remark',
        'proposal': 'Exp_proposal',
        'sampleName': 'Sample_samplename',
        'localContact': 'Exp_localcontact',
        'user': 'Exp_users',
        'scanCommand': 'info',
    }

    def __init__(self, filepath: Path, calibration_file: str | None = None):
        super().__init__(filepath, calibration_file)
        with open(filepath) as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise AttributeError(f'File "{filepath}" is empty')

        self.parameters, data_line = self.read_header(lines)
        titles, units, table = self.read_scan_data(lines, data_line)
        self._check_has_data(table)

        # Scan parameters, with the last one moved to the front (as in DataFile)
        scan = self.scan_columns(titles)
        order = [scan[-1], *scan[:-1]]
        self.scan_parameters = np.array([titles[i] for i in order])
        self.scan_units = np.array([units[i] for i in order])
        self.scan_values = table[:, order].T

        channels = self.channel_columns(titles)
        self.counts = table[:, channels].astype(int).reshape(
            -1, self.n_detectors, self.n_analysers)
        self.n_steps = self.counts.shape[0]

        self.time = self.get_column(titles, table, 'timer')
        self.monitor1 = self.get_column(titles, table, 'mon1')
        self.monitor2 = self.get_column(titles, table, 'mon2')

        self.sample = self.get_sample()
        self.A3 = self.get_scan_value(titles, table, 'sth')
        self.A3Offset = self.sample.A3Off
        self.A4 = self.get_scan_value(titles, table, 'stt')
        self.A4Offset = np.array([0.0])
        self.twotheta = self.A4 - self.A4Offset
        self.Ei = np.array([self.get_header_value('Ei_value')])
        self.get_metadata(lines[0])

    @abstractmethod
    def scan_columns(self, titles: list[str]) -> list[int]:
        """Indices of the columns holding scanned/logged parameters"""
        pass

    @classmethod
    @abstractmethod
    def channel_columns(cls, titles: list[str]) -> list[int]:
        """Indices of the detector channel columns, ordered detector-major"""
        pass

    @classmethod
    def can_read(cls, filepath: Path) -> bool:
        """A NICOS data file with this instrument's detector channels"""
        try:
            titles = cls.read_column_titles(filepath)
            return len(cls.channel_columns(titles)) == cls.n_detectors * cls.n_analysers
        except (OSError, UnicodeDecodeError, ValueError):
            return False

    @staticmethod
    def read_column_titles(filepath: Path) -> list[str]:
        """Column titles of the scan data, reading only the header of the file"""
        with open(filepath) as f:
            if 'NICOS data file' not in f.readline():
                raise ValueError(f'"{filepath}" is not a NICOS data file')
            for line in f:
                if line.find('### Scan data') > -1:
                    return [title for title in next(f, '').split()[1:] if title != ';']
        raise ValueError('No "### Scan data" section found')

    @staticmethod
    def read_header(lines: list[str]) -> tuple[dict[str, str], int]:
        """Read '#  param : value' header lines up to the scan data marker"""
        parameters = {}
        for i, line in enumerate(lines):
            if line.find('### Scan data') > -1:
                return parameters, i
            if line[:2] == '# ':
                param, _, value = line.partition(':')
                parameters[param.replace('#', '').strip()] = value.strip()
        raise AttributeError('No "### Scan data" section found')

    @staticmethod
    def read_scan_data(lines: list[str], data_line: int):
        """Return column titles, units and the numeric table of the scan"""
        titles = lines[data_line + 1].split()[1:]
        units = lines[data_line + 2].split()[1:]
        keep = [i for i, title in enumerate(titles) if title != ';']
        rows = [line.split() for line in lines[data_line + 3:]
                if line.strip() and not line.startswith('#')]
        table = np.array([[float(row[i]) for i in keep] for row in rows])
        return [titles[i] for i in keep], [units[i] for i in keep], table

    @staticmethod
    def get_column(titles: list[str], table: npt.NDArray[np.float64], name: str):
        if name not in titles:
            return None
        return table[:, titles.index(name)]

    def get_header_value(self, key: str) -> float:
        """Numeric value of a header entry such as 'Ei_value : 6.300 meV'"""
        return float(self.parameters[key].split(' ')[0])

    def get_scan_value(self, titles: list[str], table: npt.NDArray[np.float64], name: str):
        """Per-step value of a scanned axis, falling back to the header value"""
        column = self.get_column(titles, table, name)
        if column is None:
            return np.array([self.get_header_value(f'{name}_value')])
        return column

    def get_sample(self) -> Sample:
        """Build the sample and its UB matrix from the header orientation"""
        def from_parenthesis(string):
            return list(map(float, re.findall(r'(?<=\().*?(?=\))', string)[0].split(',')))

        cell = np.concatenate([from_parenthesis(self.parameters[key])
                               for key in ['Sample_lattice', 'Sample_angles']])
        q1 = np.array(from_parenthesis(self.parameters['Sample_orient1']))
        q2 = np.array(from_parenthesis(self.parameters['Sample_orient2']))

        return self.sample_from_reflections(
            cell, q1, q2, A3Offset=self.get_header_value('Sample_psi0'),
            name=self.parameters['Sample_samplename'])

    @classmethod
    def metadata_keys(cls) -> list[str]:
        return ['startTime', *cls.nicos_meta_keys]

    def get_metadata(self, first_line: str):
        # Format for starting time is 2021-11-26 10:45:05
        match = re.search(r'([0-9]+)-([0-9]+)-([0-9]+)\s([0-9]+):([0-9]+):([0-9]+)', first_line)
        if match is not None:
            self._metadata['startTime'] = \
                '-'.join(match.groups()[:3]) + ' ' + ':'.join(match.groups()[3:])
        for key in self.nicos_meta_keys:
            value = self.parameters.get(self.nicos_meta_keys[key])
            if value is not None:
                self._metadata[key] = value

    def rebin(self, binning: int, shape: tuple, bounds: npt.NDArray[np.int32]):
        # Channels are already one per analyser, so there are no pixels to sum
        return self.counts.reshape(shape).copy()
