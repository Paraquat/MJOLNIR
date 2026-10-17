"""Smoke tests for the (old) DataSet: every public operation runs on real CAMEA data and
returns something sensible. They pin behaviour while DataSet is being refactored, so they
check shapes and sanity rather than exact values."""
import pytest

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data import DataSet as DataSetModule
from MJOLNIR.Data.DataSet import DataSet

data_path = Path(__file__).parent.parent / 'sampledata'
files = [str(f) for f in sorted(data_path.glob('camea2024*.hdf'))]

# Cut parameters inside the (h, k, E) coverage of the first two files:
# h in [-1.47, 0.72], k in [-0.61, 1.54], l = 0, E in [-1.21, 0.86] meV
q1 = np.array([-1.0, 0.0, 0.0])
q2 = np.array([0.0, 1.0, 0.0])
EMin, EMax = -1.0, 0.8

@pytest.fixture(scope="module") # Convert the files once: common across all tests here
def dataset():
    ds = DataSet(files[:2])
    ds.convertDataFile()
    return ds

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

def assert_cut(data):
    assert len(data) > 0
    assert np.all(np.isfinite(data['Monitor']))
    assert np.any(data['Int'] > 0)

# Storage container

def test_container():
    ds = DataSet(files[:3])
    assert len(ds) == 3
    assert Path(ds[0].fileLocation).name == 'camea2024n001486.hdf'
    assert [Path(df.fileLocation).name for df in ds] == \
        ['camea2024n001486.hdf', 'camea2024n001487.hdf', 'camea2024n001488.hdf']
    ds.append(files[3])
    assert len(ds) == 4
    del ds[0]
    assert len(ds) == 3
    assert Path(ds[0].fileLocation).name == 'camea2024n001487.hdf'
    with pytest.raises(IndexError):
        ds[10]

def test_sorting():
    ds = DataSet(files)
    ds.autoSort()
    Ei = [np.round(df.Ei[0], 1) for df in ds]
    assert Ei == sorted(Ei)
    other = DataSet(files[::-1])
    order = ds.argSortTo(other)
    assert sorted(order) == list(range(len(files)))

# Conversion and combined data

def test_convert(dataset):
    assert len(dataset.convertedFiles) == 2
    assert dataset[0].type == 'nxs'
    assert dataset[0].I.shape == (135, 104, 64)
    for name in ['I', 'qx', 'qy', 'energy', 'Norm', 'Monitor']:
        values = getattr(dataset, name).extractData()
        assert values.shape == dataset.I.extractData().shape
        assert np.all(np.isfinite(values))

@pytest.mark.xfail(strict=True, reason='Old DataSet points h/k/l at attributes H/K/L, '
                   'but converted files store them as h/k/l')
def test_combined_hkl(dataset):
    dataset.h.extractData()

def test_coordinates(dataset):
    QxQy = dataset.convertToQxQy(np.array([q1, q2]))
    assert np.allclose(dataset.convertToHKL(QxQy), [q1, q2])

def test_mask():
    ds = DataSet(files[:2])
    ds.convertDataFile()
    total = sum(df.I.size for df in ds)
    mask = [np.zeros_like(df.I, dtype=bool) for df in ds]
    mask[0][:, 0] = True
    ds.mask = mask # Replaces the default mask set during conversion
    assert ds.I.extractData().size == total - mask[0].sum()
    ds.appendMask([np.ones_like(m) if i == 1 else np.zeros_like(m) for i, m in enumerate(mask)])
    assert ds.I.extractData().size == ds[0].I.size - mask[0].sum()
    ds.calculateCurratAxeMask(BraggPeaks=[[0, 1, 0]], dqx=0.05, dqy=0.05)

@pytest.mark.xfail(strict=True, reason='Old removeMasking reads self._maskingObject, which '
                   'only exists after a Mask object has been applied')
def test_removeMasking():
    ds = DataSet(files[:1])
    ds.convertDataFile()
    ds.removeMasking()

def test_subtraction(dataset):
    difference = dataset - dataset
    assert len(difference) == len(dataset)
    assert np.all(difference.I.extractData() == 0)

# Cuts

def test_cut1D(dataset):
    data, bins = dataset.cut1D(q1, q2, width=0.1, minPixel=0.02, EMin=EMin, EMax=EMax)
    assert_cut(data)

def test_cut1DE(dataset):
    data, bins = dataset.cut1DE(E1=EMin, E2=EMax, q=np.array([-0.5, 0.5, 0.0]),
                                width=0.1, minPixel=0.05)
    assert_cut(data)

def test_cutQE(dataset):
    data, positions, centers = dataset.cutQE(
        q1, q2, width=0.1, minPixel=0.02, EMin=EMin, EMax=EMax, dE=0.1)
    assert_cut(data)

def test_cutQELine(dataset):
    data, bins = dataset.cutQELine(QPoints=np.array([q1, [0.0, 0.0, 0.0], q2]),
                                   EnergyBins=np.linspace(EMin, EMax, 10))
    assert len(data) == 2 # One cut per segment
    for segment in data:
        assert_cut(segment)

CUT_E_LINE_BUG = 'Old cutELine names its parameter EMn instead of EMin: EMin is always unset'

@pytest.mark.xfail(strict=True, reason=CUT_E_LINE_BUG)
def test_cutELine(dataset):
    data, bins = dataset.cutELine(q1, q2, EMax=EMax, energyWidth=0.1, minPixel=0.05)
    assert_cut(data)

def test_cutPowder(dataset):
    data = dataset.cutPowder(EBins=np.linspace(EMin, EMax, 10), QBins=np.linspace(0, 2, 20))
    assert_cut(data)

def test_binData3D(dataset):
    data, bins = dataset.binData3D(0.05, 0.05, 0.1)
    assert len(bins) == 3
    assert np.nansum(data[0]) > 0

def test_cutRaw1D(dataset):
    data = dataset.cutRaw1D()
    assert len(data) > 0

# Plots

def test_plotCut1D(dataset):
    ax, data, bins = dataset.plotCut1D(q1, q2, width=0.1, minPixel=0.02, EMin=EMin, EMax=EMax)
    assert_cut(data)

def test_plotCut1DE(dataset):
    ax, data, bins = dataset.plotCut1DE(E1=EMin, E2=EMax, q=np.array([-0.5, 0.5, 0.0]),
                                        width=0.1, minPixel=0.05)
    assert_cut(data)

def test_plotCutQE(dataset):
    ax, data, bins = dataset.plotCutQE(q1, q2, EMin=EMin, EMax=EMax, dE=0.1,
                                       width=0.1, minPixel=0.02, colorbar=True)
    assert_cut(data)

def test_plotCutQELine(dataset):
    ax, data, *bins = dataset.plotCutQELine(QPoints=np.array([q1, [0.0, 0.0, 0.0], q2]),
                                            EnergyBins=np.linspace(EMin, EMax, 10))
    assert len(data) > 0
    assert np.nansum([np.nansum(segment['Int']) for segment in data]) > 0

@pytest.mark.xfail(strict=True, reason=CUT_E_LINE_BUG)
def test_plotCutELine(dataset):
    ax, data, bins = dataset.plotCutELine(q1, q2, EMax=EMax, energyWidth=0.1, minPixel=0.05)
    assert_cut(data)

def test_plotCutPowder(dataset):
    ax, data, centersQ, centersE = dataset.plotCutPowder(EBins=np.linspace(EMin, EMax, 10),
                                     QBins=np.linspace(0, 2, 20))
    assert_cut(data)

def test_plotQPlane(dataset):
    *_, ax = dataset.plotQPlane(EMin=-0.2, EMax=0.2, xBinTolerance=0.03,
                                yBinTolerance=0.03, colorbar=True)
    assert ax is not None

@pytest.mark.xfail(strict=True, reason='Old plotA3A4 calls voronoiTessellation, '
                   'which no longer exists')
def test_plotA3A4(dataset):
    dataset.plotA3A4()

def test_plotRaw1D(dataset):
    dataset.plotRaw1D()

def test_View3D(dataset):
    viewer = dataset.View3D(0.05, 0.05, 0.1)
    assert viewer is not None

# Saving

@pytest.mark.xfail(strict=True, reason='Old DataSet cannot be pickled: its PointerArrays '
                   'hold a lambda')
def test_save_load(dataset, tmp_path):
    dataset.save(tmp_path / 'dataset.pkl')
    loaded = DataSetModule.load(tmp_path / 'dataset.pkl')
    assert len(loaded) == len(dataset)
