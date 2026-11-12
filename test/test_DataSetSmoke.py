"""Smoke tests for DataSet: every public operation runs on real CAMEA data and returns something
sensible (shapes and sanity, not exact values: those are in the other test files and in
test_Reference.py, against the old code). test_every_operation_is_run checks that no public
DataSet method is missing here."""
import pytest

import warnings
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data import Mask
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.CutResult import CutResult
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

ROOT = Path(__file__).parent.parent
files = sorted((ROOT / 'sampledata').glob('camea2024*.hdf'))
pytestmark = pytest.mark.skipif(len(files) < 4, reason='sampledata/camea2024*.hdf not available')

# Cut parameters inside the (h, k, E) coverage of the first two files:
# h in [-1.47, 0.72], k in [-0.61, 1.54], l = 0, E in [-1.21, 0.86] meV
q1, q2 = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
energy_min, energy_max = -1.0, 0.8
energy_bins = np.linspace(-1.0, 0.8, 7)

def new(count: int = 2) -> DataSet:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return DataSet(InstrumentName.CAMEA, files=files[:count])

@pytest.fixture(scope='module') # Read the files once, for the operations that change nothing
def dataset():
    return new()

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

def assert_cut(result: CutResult):
    assert isinstance(result, CutResult) and len(result.data) > 0
    assert np.all(np.isfinite(result.data['Monitor']))
    assert np.any(result.data['Int'] > 0)

def test_every_operation_is_run():
    source = Path(__file__).read_text()
    public = [name for name in vars(DataSet) if not name.startswith('_')]
    assert [name for name in public if f'.{name}' not in source] == []

# Container

def test_container():
    ds = new()
    assert len(ds) == ds.number_of_files() == 2
    assert [reader.file.name for reader in ds] == [f.name for f in files[:2]]
    assert ds[0].file == files[0] and isinstance(ds[0:1], DataSet) and len(ds[0:1]) == 1
    assert repr(ds) == f'DataSet([{files[0].name}, {files[1].name}])'
    ds.append(files[2])
    ds.add_file(files[3])
    assert len(ds) == 4
    del ds[2:]
    ds.extend(files[2:4])
    del ds[3]
    ds.add_files([files[3]])
    assert [reader.file.name for reader in ds] == [f.name for f in files[:4]]
    assert ds.instrument is InstrumentName.CAMEA

def test_sorting():
    ds = new(4)
    ds.apply_sorting([3, 1, 2, 0])
    keys = [DataSet.default_sort_key(reader) for reader in ds]
    order = ds.arg_auto_sort()
    assert sorted(keys) == [keys[i] for i in order]
    ds.auto_sort()
    assert [DataSet.default_sort_key(reader) for reader in ds] == sorted(keys)
    other = new(4)
    other.apply_sorting([2, 0, 3, 1])
    other.apply_sorting(other.arg_sort_to(ds))
    assert [reader.file for reader in other] == [reader.file for reader in ds]

def test_binning():
    ds = new(1)
    assert ds[0].binning == 8 and ds[0].intensity.shape[-1] == 64
    ds.set_binning(3)
    assert ds[0].binning == 3 and ds[0].intensity.shape[-1] == 24

def test_coordinates(dataset):
    hkl = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    assert np.allclose(dataset.convert_to_hkl(dataset.convert_to_qxqy(hkl)), hkl)
    qx, qy = dataset.gather_q()
    assert len(np.asarray(qx)) == len(np.asarray(qy)) == len(np.asarray(dataset.gather(ArrayType.QX)))
    assert len(dataset.rotations()) == 2 and dataset.sample is dataset.samples[0]
    assert dataset.get_view(ArrayType.INTENSITY).arrays[0].shape == dataset[0].intensity.shape

def test_mask():
    ds = new()
    assert ds.use_default_mask
    ds.mask = [np.zeros(reader.intensity.shape, dtype=bool) for reader in ds]
    ds.append_mask([np.ones_like(m) if i == 1 else np.zeros_like(m) for i, m in enumerate(ds.mask)])
    assert ds[1].user_mask.all() and not ds[0].user_mask.any()
    ds.mask = Mask.CurratAxeMask([[0, 1, 0]], dqx=0.05, dqy=0.05) # A mask object
    ds.remove_masking()
    assert not any(reader.user_mask.any() for reader in ds)
    ds.use_default_mask = False
    assert not any(reader.mask.any() for reader in ds)

def test_subtraction(dataset):
    difference = dataset - dataset
    assert difference.frozen and len(difference) == 2
    assert np.allclose(difference[0].intensity, 0.0)

# Cuts

def test_cut_1d(dataset):
    assert_cut(dataset.cut_1d(q1, q2, 0.1, 0.02, energy_min, energy_max))

def test_cut_constant_q(dataset):
    assert_cut(dataset.cut_constant_q(q1, width=0.1, min_pixel=0.1))

def test_cut_qe(dataset):
    assert_cut(dataset.cut_qe(q1, q2, 0.1, 0.05, energy_bins=energy_bins))

def test_cut_qe_line(dataset):
    result = dataset.cut_qe_line([q1, q2, [0.5, 0.5, 0.0]], energy_bins)
    assert_cut(result)
    assert sorted(result.data['qCut'].unique()) == [0, 1]

def test_cut_powder(dataset):
    result = dataset.cut_powder(np.arange(0.4, 2.0, 0.05), energy_bins)
    assert_cut(result)
    assert result.grid('Int').shape == (31, 6)

def test_cut_q_plane(dataset):
    assert_cut(dataset.cut_q_plane(energy_min=-0.2, energy_max=0.2))

def test_bin_3d(dataset):
    result = dataset.bin_3d(0.05, 0.05, 0.2)
    assert_cut(result)
    assert result.grid('Int').ndim == 3

def test_cut_raw_1d(dataset):
    results = dataset.cut_raw_1d()
    assert len(results) == 2 and all(np.any(r.data['intensity'] > 0) for r in results)

# Plots

def test_plots(dataset):
    assert dataset.plot_cut_1d(q1, q2, 0.1, 0.02, energy_min, energy_max).Data is not None
    assert dataset.plot_cut_constant_q(q1, width=0.1, min_pixel=0.1).type == 'ConstantQ'
    assert dataset.plot_cut_qe(q1, q2, energy_bins=energy_bins).type == 'QE'
    assert dataset.plot_cut_qe_line([q1, q2, [0.5, 0.5, 0.0]], energy_bins).type == 'QELine'
    assert dataset.plot_cut_powder(np.arange(0.4, 2.0, 0.05), energy_bins).type == 'Powder'
    assert dataset.plot_q_plane(energy_min=-0.2, energy_max=0.2).type == 'QPlane'
    assert len(dataset.plot_raw_1d().results) == 2
    viewer = dataset.view_3d(0.05, 0.05, 0.2, on_click=None)
    assert viewer.values.ndim == 3
    assert dataset.create_q_axis().get_xlabel().endswith('[RLU]')
    assert dataset.create_qe_axes(axis=0) is not None

# Backgrounds

def test_background():
    ds = new()
    model = ds.generate_background_model()
    assert model is ds.background_model and all(r.background is not None for r in ds)
    assert_cut(ds.cut_1d(q1, q2, 0.1, 0.02, energy_min, energy_max, background_subtraction=True))
    ds.remove_background_model()
    assert ds.background_model is None and all(r.background is None for r in ds)

def test_amber_background():
    pytest.importorskip('AMBER')
    ds = new(1)
    model = ds.generate_amber_background(n_epochs=1, verbose=False)
    assert model is ds.background_model and ds[0].background is not None

# Calibration, sample, normalisation

def test_update_calibration():
    ds = new(1)
    assert ds.update_calibration(ROOT / 'Data' / 'Normalization_8.calib', overwrite=True) == [8]

def test_update_sample_parameters():
    ds = new(1)
    cell = ds.sample.unitCell*[1.01, 1.0, 1.0, 1.0, 1.0, 1.0]
    ds.update_sample_parameters(cell)
    assert np.allclose(ds.sample.unitCell, cell)

def test_absolute_normalize():
    ds = new(1)
    factor = ds.absolute_normalize(2.0, 'TbFeO3')
    assert factor > 0 and ds.absolute_normalization == factor
    ds.undo_absolute_normalize()
    assert ds.absolute_normalization == 1.0

# Export and physics

def test_generate_ufit_dataset(dataset):
    pytest.importorskip('ufit')
    result = dataset.cut_1d(q1, q2, 0.1, 0.02, energy_min, energy_max)
    assert len(dataset.generate_ufit_dataset(result).x) == len(result.data)

def test_physics(dataset):
    assert np.allclose(dataset.find_ei(0.0), [4.0])
    assert np.isnan(dataset.second_order_elastic([0.5])).all()
    M = dataset.resolution_matrix(q1, 4.0, 3.5)
    assert M.shape == (4, 4)
    _, axes, sigma = dataset.resolution_ellipse(q1, [1, 0, 0, 0], [0, 0, 0, 1], 4.0, 3.5)
    assert axes.shape == (2, 2) and np.all(sigma > 0)
