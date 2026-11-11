from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Names of the columns of cut results, as in the old DataSet (pdNaming)
COLUMNS = {
    'qx': 'Qx',
    'qy': 'Qy',
    'h': 'H',
    'k': 'K',
    'l': 'L',
    'qLength': 'q', # |Q| of powder cuts
    'e': 'Energy',
    'int': 'Int',
    'intensity': 'intensity',
    'intError': 'Int_err',
    'mon': 'Monitor',
    'norm': 'Normalization',
    'binCount': 'BinCount',
    'Foreground': 'Int_Fg',
    'ForegroundIntensity': 'intensity_Fg',
    'BackgroundIntensity': 'intensity_Bg',
    'Background': 'Int_Bg',
    'ForegroundError': 'Int_Fg_err',
    'BackgroundError': 'Int_Bg_err',
    'plotPosition': 'BinDistance',
}

@dataclass
class CutResult:
    """Result of a cut: the binned data, and where the bins are.

    data has one row per bin. Its columns (named as in the old DataSet, see COLUMNS):

        - Qx, Qy: Bin position in the frame of the data set [1/AA]
        - H, K, L: Bin position in reciprocal lattice units (NaN without a sample)
        - Energy: Energy transfer of the bin [meV]
        - intensity: Neutron counts summed over the bin
        - Monitor, Normalization: Monitor and normalization summed over the bin
        - BinCount: Number of data points in the bin
        - Int: Normalised intensity, intensity * BinCount / (Monitor * Normalization)
        - Int_err: Error of Int, from the counting statistics

    bins holds the bin edges by name, depending on the kind of cut; parameters the arguments
    the cut was made with.
    """
    data: pd.DataFrame
    bins: dict[str, np.ndarray] = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data)

    def grid(self, column: str) -> np.ndarray:
        """Values of a column (e.g. 'Int') in the shape of the bins, for a result binned on a
        regular grid (Cuts.bin_3d): one axis per name in parameters['grid'], whose edges are in
        bins"""
        if 'grid' not in self.parameters:
            raise ValueError('This result is not binned on a regular grid')
        shape = tuple(len(self.bins[axis]) - 1 for axis in self.parameters['grid'])
        return self.data[column].to_numpy().reshape(shape)
