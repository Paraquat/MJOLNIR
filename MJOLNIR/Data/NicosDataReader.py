import re
import numpy as np
from pathlib import Path
from abc import abstractmethod

from MJOLNIR import TasUBlibDEG as TasUBlib
from MJOLNIR.Data.DataReader import DataReader, InstrumentCalibration
from MJOLNIR.Data.Sample import Sample

class NICOSDataReader(DataReader):
    """Parse a NICOS ASCII output file"""

    # Location of the instrument calibration table, set by subclasses
    calibration_file: str

    # NICOS ASCII files contain a single calibration, with one bin per analyser
    possible_binnings = [1]

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
        super().__init__(filepath)
        if calibration_file is not None:
            self.calibration_file = calibration_file
        with open(filepath) as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise AttributeError(f'File "{filepath}" is empty')

        self.parameters, data_line = self.read_header(lines)
        titles, units, table = self.read_scan_data(lines, data_line)

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

        self.A3 = self.get_scan_value(titles, table, 'sth')
        self.A3Offset = np.array([0.0])
        self.A4 = self.get_scan_value(titles, table, 'stt')
        self.A4Offset = np.array([0.0])
        self.twotheta = self.A4 - self.A4Offset
        self.Ei = np.array([self.get_header_value('Ei_value')])
        self.sample = self.get_sample()
        self.get_metadata(lines[0])

    @abstractmethod
    def scan_columns(self, titles: list[str]) -> list[int]:
        """Indices of the columns holding scanned/logged parameters"""
        pass

    @abstractmethod
    def channel_columns(self, titles: list[str]) -> list[int]:
        """Indices of the detector channel columns, ordered detector-major"""
        pass

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
    def get_column(titles: list[str], table: np.NDArray[np.float64], name: str):
        if name not in titles:
            return None
        return table[:, titles.index(name)]

    def get_header_value(self, key: str) -> float:
        """Numeric value of a header entry such as 'Ei_value : 6.300 meV'"""
        return float(self.parameters[key].split(' ')[0])

    def get_scan_value(self, titles: list[str], table: np.NDArray[np.float64], name: str):
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

        Ei = 10.0 # TODO: Dummy incoming energy needed to calculate UB (from DataFile)
        k = np.sqrt(Ei) * TasUBlib.factorsqrtEK
        B = TasUBlib.calculateBMatrix(TasUBlib.calcCell(cell))
        A3Offset = self.get_header_value('Sample_psi0')

        A41 = TasUBlib.calTwoTheta(B, [*q1, Ei, Ei], -1)
        A31 = TasUBlib.calcTheta(k, k, A41) + A3Offset
        A42 = TasUBlib.calTwoTheta(B, [*q2, Ei, Ei], -1)
        A32 = TasUBlib.calcTheta(k, k, A42)

        # h, k, l, A3, A4, sgu, sgl, Ei, Ef
        plane_vector1 = np.array([*q1, A31, A41, 0.0, 0.0, Ei, Ei])
        plane_vector2 = np.array([*q2, A32, A42, 0.0, 0.0, Ei, Ei])
        plane_vector2[3] += TasUBlib.tasAngleBetweenReflections(
            B, plane_vector1, plane_vector2)

        a, b, c, alpha, beta, gamma = cell
        return Sample(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                      projectionVector1=plane_vector1, projectionVector2=plane_vector2,
                      name=self.parameters['Sample_samplename'])

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

    def get_instrument_calibration(self, binning: int) -> InstrumentCalibration:
        if binning not in self.possible_binnings:
            raise AttributeError(
                f'Binning {binning} not in possible binnings {self.possible_binnings}')
        # Columns: detector, energy, pixel, amplitude, center, width, background,
        # lowerBin, upperBin, A4Offset
        table = np.genfromtxt(self.calibration_file, skip_header=1, delimiter=',')
        Ef = table[:, 4]
        width = table[:, 5]
        bg = table[:, 6]
        amp = table[:, 3]
        A4 = -table[:, 9]
        bound = table[:, 7:9].astype(int)
        self.binning = binning
        return InstrumentCalibration(Ef, width, bg, amp, A4, bound)

    def set_instrument_calibration(self, binning: int = 1):
        self.calibration = self.get_instrument_calibration(binning)

    def rebin(self, binning: int, shape: tuple, bounds: np.NDArray[np.int32]):
        # Counts are already given per analyser, so there are no pixels to sum
        return self.counts.reshape(shape).copy()
