import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data import Mask
from MJOLNIR.Data.DataSet import DataSet, ArrayType
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.DataReader import DataReader
from MJOLNIR.Data.InstrumentName import InstrumentName

data_path = Path(__file__).parent.parent / 'sampledata'
files = sorted(data_path.glob('camea2024*.hdf'))

@pytest.fixture(scope="module") # Read the files: common across all test functions here
def readers():
    return [CameaDataReader(f) for f in files]

def names(dataset):
    return [reader.file.name for reader in dataset]


def test_dataset():
    dataset = DataSet(InstrumentName.CAMEA)
    dataset.set_binning(3)
    dataset.add_files(files)
    assert dataset.number_of_files() == 8
    first_reader = dataset._readers[0]
    second_reader = dataset._readers[1]
    assert first_reader.file.name == 'camea2024n001486.hdf'
    assert second_reader.file.name == 'camea2024n001487.hdf'
    assert first_reader.counts.shape == (135, 104, 1024)
    assert first_reader.A4[0] == pytest.approx(-43.99, abs=1e-2)
    assert first_reader.Ei == pytest.approx(4.00048, abs=1e-5)
    assert first_reader._metadata['proposal'] == '20240261'
    assert first_reader._metadata['sampleName'] == 'TbFeO3'

    ref_h_1 = dataset._readers[0].h[134, 0, 0]
    ref_h_2 = dataset._readers[1].h[0, 0, 0]
    h_view = dataset.get_view(ArrayType.H)

    # array shape
    assert len(h_view) == 1080
    assert h_view.shape[0] == 1080
    assert h_view.shape[1] == 104
    assert h_view.shape[2] == 24

    # contiguity
    assert h_view[134, 0, 0] == pytest.approx(ref_h_1)
    assert h_view[135, 0, 0] == pytest.approx(ref_h_2)

def test_create_from_paths():
    dataset = DataSet(files=files[:3], instrument=InstrumentName.CAMEA)
    assert len(dataset) == 3
    assert all(isinstance(reader, CameaDataReader) for reader in dataset)
    assert names(dataset) == \
        ['camea2024n001486.hdf', 'camea2024n001487.hdf', 'camea2024n001488.hdf']
    assert dataset.sample.a == pytest.approx(5.34993, abs=1e-5)

def test_create_from_readers(readers):
    assert len(DataSet()) == 0
    assert DataSet().sample is None
    assert names(DataSet(files=readers[0])) == ['camea2024n001486.hdf']
    dataset = DataSet(files=readers)
    assert len(dataset) == 8
    assert dataset[0] is readers[0]

def test_path_needs_instrument():
    with pytest.raises(ValueError):
        DataSet(files=files[0])
    with pytest.raises(TypeError):
        DataSet(files=[42])

def test_indexing(readers):
    dataset = DataSet(files=readers[:4])
    assert dataset[-1] is readers[3]
    with pytest.raises(IndexError):
        dataset[4]
    subset = dataset[1:3]
    assert isinstance(subset, DataSet)
    assert names(subset) == ['camea2024n001487.hdf', 'camea2024n001488.hdf']
    assert names(reversed(dataset)) == names(dataset)[::-1]
    assert repr(dataset).startswith('DataSet([camea2024n001486.hdf, ')

def test_nested_iteration(readers):
    dataset = DataSet(files=readers[:3])
    pairs = [(a.file.name, b.file.name) for a in dataset for b in dataset]
    assert len(pairs) == 9

def test_append_extend_delete(readers):
    dataset = DataSet(files=readers[:2], instrument=InstrumentName.CAMEA)
    dataset.append(readers[2])
    dataset.append(files[3]) # Paths are read with the data set's instrument
    dataset.extend(readers[4:6])
    assert len(dataset) == 6
    assert isinstance(dataset[3], CameaDataReader)
    del dataset[0]
    del dataset[-1]
    assert names(dataset) == [f.name for f in files[1:5]]
    del dataset[1:3]
    assert names(dataset) == [files[1].name, files[4].name]
    with pytest.raises(IndexError):
        del dataset[2]

def test_sample_must_match(readers):
    dataset = DataSet(files=readers[:1])
    other = CameaDataReader(files[1])
    other.sample = DataReader.sample_from_reflections(
        [4.0, 4.0, 4.0, 90, 90, 90], np.array([1, 0, 0]), np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        dataset.append(other)
    assert len(dataset) == 1

class OtherReader(CameaDataReader):
    """A different reader type for the same files"""

def test_single_reader_type(readers):
    dataset = DataSet(files=readers[:2])
    assert dataset.reader_type is CameaDataReader
    # Instrument is taken from the reader type, so paths can be added
    assert dataset.instrument == InstrumentName.CAMEA
    dataset.append(files[2])
    with pytest.raises(TypeError):
        dataset.append(OtherReader(files[3]))
    assert len(dataset) == 3

    with pytest.raises(TypeError):
        DataSet(files=[readers[0], OtherReader(files[1])])
    with pytest.raises(TypeError):
        DataSet(files=OtherReader(files[0]), instrument=InstrumentName.CAMEA)
    with pytest.raises(TypeError):
        DataSet(files=readers[0], instrument=InstrumentName.BAMBUS)

    empty = DataSet(instrument=InstrumentName.CAMEA)
    assert empty.reader_type is CameaDataReader
    assert dataset[5:].reader_type is CameaDataReader

    other = DataSet(files=[OtherReader(f) for f in files[:2]])
    assert other.reader_type is OtherReader
    assert other.instrument is None # Not registered with the factory
    with pytest.raises(ValueError):
        other.append(files[2])

def test_sorting(readers):
    dataset = DataSet(files=readers)
    order = dataset.arg_auto_sort()
    # Same order as the old DataSet.argAutoSort for these files
    assert list(order) == [1, 0, 5, 4, 2, 3, 6, 7]
    dataset.auto_sort()
    assert names(dataset) == [files[i].name for i in order]
    Ei = [np.round(reader.Ei[0], 1) for reader in dataset]
    assert Ei == sorted(Ei)

    by_A4 = DataSet(files=readers)
    by_A4.auto_sort(key=lambda reader: (reader.A4[0],))
    assert [reader.A4[0] for reader in by_A4] == sorted(reader.A4[0] for reader in readers)

def test_sort_to(readers):
    dataset = DataSet(files=readers)
    other = DataSet(files=readers[::-1])
    order = dataset.arg_sort_to(other)
    dataset.apply_sorting(order)
    assert names(dataset) == names(other)
    with pytest.raises(ValueError):
        dataset.apply_sorting([0, 0, 1, 2, 3, 4, 5, 6])

def test_view_per_file_values(readers):
    dataset = DataSet(files=readers)
    # A4 holds one value per file
    A4 = dataset.get_view(ArrayType.A4)
    assert len(A4) == 8
    assert A4[1,] == pytest.approx(readers[1].A4[0])
    assert [a4 for a4 in A4] == pytest.approx([reader.A4[0] for reader in readers])

def test_view_per_step_values(readers):
    dataset = DataSet(files=readers[:3])
    monitor = dataset.get_view(ArrayType.MONITOR1)
    assert monitor.shape == (405,)
    # Entry 135 is the first scan step of the second file
    assert monitor[134,] == readers[0].monitor1[-1]
    assert monitor[135,] == readers[1].monitor1[0]
    assert monitor[-1,] == readers[2].monitor1[-1]
    assert np.array_equal(monitor[130:140,], np.concatenate([readers[0].monitor1[130:],
                                                            readers[1].monitor1[:5]]))
    counts = dataset.get_view(ArrayType.COUNTS)
    assert counts.shape == (405, 104, 1024)
    assert counts[135, 0, 93] == readers[1].counts[0, 0, 93] == 1

def test_view_calibration(readers):
    dataset = DataSet(files=readers[:2])
    Ef = dataset.get_view(ArrayType.EF_CAL)
    # Default CAMEA binning 8: 8 energies x 8 bins for each of 104 detectors
    assert len(Ef) == 2 * 104 * 8 * 8
    assert Ef[6656,] == pytest.approx(readers[1].calibration.Ef[0])
    assert dataset.get_view(ArrayType.BOUND_CAL).shape == (13312, 2)

def test_set_binning():
    dataset = DataSet(InstrumentName.CAMEA, files=files[:2])
    assert [reader.binning for reader in dataset] == [8, 8] # Default CAMEA binning
    assert dataset.get_view(ArrayType.QX).shape == (270, 104, 64)
    dataset.set_binning(3)
    # Data files are transformed again, so derived arrays follow the binning
    assert dataset.get_view(ArrayType.QX).shape == (270, 104, 24)
    assert dataset.get_view(ArrayType.INTENSITY).shape == (270, 104, 24)
    dataset.append(CameaDataReader(files[2])) # Added files get the binning of the DataSet
    assert dataset[2].binning == 3

def test_view_empty():
    with pytest.raises(ValueError):
        DataSet(InstrumentName.CAMEA).get_view(ArrayType.A4)

# Masking: each test makes its own DataSet, so masks do not leak into the shared readers

@pytest.fixture
def masked_dataset():
    # These tests are about the user's mask, so the instrument's default mask is switched off
    # (it is tested in test_DefaultMask.py)
    dataset = DataSet(InstrumentName.CAMEA, files=files[:2])
    dataset.use_default_mask = False
    return dataset

circle = Mask.circleMask(center=[-1, 0], radius=0.25, coordinates=['h', 'k'])
low_energy = Mask.lineMask(start=0.0, end=0.5, coordinates='energy')

def test_default_mask(masked_dataset):
    masks = masked_dataset.mask
    assert len(masks) == 2
    assert all(m.shape == (135, 104, 64) and m.dtype == bool and not m.any() for m in masks)
    assert masked_dataset.get_view(ArrayType.MASK).shape == (270, 104, 64)

def test_mask_from_arrays(masked_dataset):
    shape = masked_dataset[0].intensity.shape
    first = np.zeros(shape, dtype=bool)
    first[:, 0] = True
    masked_dataset.mask = [first, np.zeros(shape, dtype=bool)] # One array per file
    assert masked_dataset.mask[0].sum() == 135 * 64
    assert not masked_dataset.mask[1].any()
    masked_dataset.mask = first # The same array for every file
    assert all(m.sum() == 135 * 64 for m in masked_dataset.mask)
    masked_dataset.mask = np.array([first, ~first]) # Stacked, one per file
    assert masked_dataset.mask[1].sum() == first.size - 135 * 64
    with pytest.raises(ValueError): # Wrong number of masks
        masked_dataset.mask = [first]
    with pytest.raises(ValueError): # Wrong shape
        masked_dataset.mask = np.zeros((135, 104, 8), dtype=bool)
    with pytest.warns(UserWarning, match='every point'):
        masked_dataset.mask = np.ones(shape, dtype=bool)

def test_mask_from_object(masked_dataset):
    mask = circle * low_energy
    arrays = mask(masked_dataset)
    masked_dataset.mask = mask
    assert all(np.array_equal(m, a) for m, a in zip(masked_dataset.mask, arrays))
    assert sum(m.sum() for m in arrays) > 0
    masked_dataset.append(files[2]) # Files added later are masked too
    assert np.array_equal(masked_dataset.mask[2], mask(masked_dataset[2:])[0])
    assert masked_dataset.mask[2].any()

def test_append_mask(masked_dataset):
    masked_dataset.mask = circle
    before = [m.copy() for m in masked_dataset.mask]
    masked_dataset.append_mask(low_energy)
    expected = [b | e for b, e in zip(before, low_energy(masked_dataset))]
    assert all(np.array_equal(m, e) for m, e in zip(masked_dataset.mask, expected))
    # Both mask objects are kept, so they follow a change of binning
    masked_dataset.set_binning(3)
    expected = [c | e for c, e in zip(circle(masked_dataset), low_energy(masked_dataset))]
    assert all(np.array_equal(m, e) for m, e in zip(masked_dataset.mask, expected))

def test_remove_masking(masked_dataset):
    masked_dataset.mask = circle
    masked_dataset.remove_masking()
    assert not any(m.any() for m in masked_dataset.mask)
    masked_dataset.set_binning(3) # Nothing comes back
    assert not any(m.any() for m in masked_dataset.mask)

def test_mask_follows_binning(masked_dataset):
    masked_dataset.mask = circle
    masked_dataset.set_binning(3)
    assert masked_dataset.mask[0].shape == (135, 104, 24)
    assert all(np.array_equal(m, c) for m, c in zip(masked_dataset.mask, circle(masked_dataset)))
    # Arrays cannot be re-evaluated: removed, with a warning
    masked_dataset.mask = [m.copy() for m in masked_dataset.mask]
    with pytest.warns(UserWarning, match='removed'):
        masked_dataset.set_binning(8)
    assert not any(m.any() for m in masked_dataset.mask)


# Gathering the unmasked points across files

def test_gather():
    dataset = DataSet(InstrumentName.CAMEA, files=files[:2]) # Binning 8: 28080 points masked per file
    qx = dataset.gather(ArrayType.QX)
    unmasked = 135 * 104 * 64 - 135 * 104 * 2
    assert len(qx) == 2 * unmasked
    assert qx.offsets == [0, unmasked] # Boundary between the files
    for reader, per_file in zip(dataset, qx.arrays):
        assert np.array_equal(per_file, reader.qx[~reader.mask])
    joined = np.asarray(qx)
    assert joined.shape == (2 * unmasked,)
    assert np.asarray(dataset.gather('intensity')).dtype == dataset[0].intensity.dtype
    assert np.max(dataset.gather(ArrayType.ENERGY)) == max(r.energy[~r.mask].max() for r in dataset)

def test_gather_follows_masks():
    dataset = DataSet(InstrumentName.CAMEA, files=files[:1])
    total = dataset[0].intensity.size
    assert len(dataset.gather(ArrayType.QX)) == total - 135 * 104 * 2 # Default mask
    dataset.mask = Mask.lineMask(start=0.0, end=0.5, coordinates='energy')
    assert len(dataset.gather(ArrayType.QX)) == total - dataset.mask[0].sum()
    dataset.use_default_mask = False
    dataset.remove_masking()
    assert len(dataset.gather(ArrayType.QX)) == total

def test_gather_errors():
    dataset = DataSet(InstrumentName.CAMEA, files=files[:1])
    with pytest.raises(ValueError, match='use get_view'): # One value per scan step
        dataset.gather(ArrayType.A3)
    with pytest.raises(ValueError):
        DataSet(InstrumentName.CAMEA).gather(ArrayType.QX)

# Samples and coordinates

def test_samples():
    dataset = DataSet(InstrumentName.CAMEA, files=files[:3])
    assert len(dataset.samples) == 3
    assert all(sample is reader.sample for sample, reader in zip(dataset.samples, dataset))
    assert dataset.sample is dataset.samples[0]

def test_coordinate_conversion():
    dataset = DataSet(InstrumentName.CAMEA, files=files[:1])
    hkl = np.array([[-0.5, 1.5, 0.0], [1.0, 0.0, 0.0]])
    qxqy = dataset.convert_to_qxqy(hkl)
    assert qxqy.shape == (2, 2)
    assert np.allclose(dataset.convert_to_qxqy(hkl[1]), qxqy[1]) # A single point
    assert np.allclose(dataset.convert_to_hkl(qxqy), hkl) # Round trip
    assert np.linalg.norm(qxqy[1]) == pytest.approx(2 * np.pi / dataset.sample.a, rel=1e-6)
    with pytest.raises(AttributeError):
        dataset.convert_to_qxqy([1, 0])
    with pytest.raises(AttributeError):
        dataset.convert_to_hkl([1, 0, 0])

def test_gather_q():
    from MJOLNIR import _tools
    dataset = DataSet(InstrumentName.CAMEA, files=files[:2])
    qx, qy = dataset.gather(ArrayType.QX), dataset.gather(ArrayType.QY)
    # Same orientation in both files: nothing to rotate
    rotated_qx, rotated_qy = dataset.gather_q()
    assert np.allclose(np.asarray(rotated_qx), np.asarray(qx))
    assert rotated_qx.offsets == qx.offsets
    # Give the second file another orientation: its points are rotated by the difference
    second = dataset.samples[1]
    second.RotMat = _tools.Rot(second.theta + 10) # Degrees, as Sample uses it
    rotated_qx, rotated_qy = dataset.gather_q()
    expected = _tools.Rot(10) @ np.array([qx.arrays[1], qy.arrays[1]])
    assert np.allclose(rotated_qx.arrays[1], expected[0])
    assert np.allclose(rotated_qy.arrays[1], expected[1])
    assert np.allclose(rotated_qx.arrays[0], qx.arrays[0]) # The first file defines the frame

def test_gather_q_aligned_to_lattice():
    dataset = DataSet(InstrumentName.CAMEA, files=files[:2])
    qx, qy = dataset.gather(ArrayType.QX), dataset.gather(ArrayType.QY)
    aligned_x, aligned_y = dataset.gather_q(align_to_lattice=True)
    # Each file rotated by its own sample's RotMat
    for sample, x, y, ax_, ay_ in zip(dataset.samples, qx.arrays, qy.arrays,
                                      aligned_x.arrays, aligned_y.arrays):
        expected = sample.RotMat @ np.array([x, y])
        assert np.allclose(ax_, expected[0]) and np.allclose(ay_, expected[1])
    # The matching sample gives the same H, K, L as the unaligned points with the original
    from MJOLNIR.Data.Coordinates import align_to_lattice, convert_to_hkl
    points = np.array([np.asarray(qx)[:5], np.asarray(qy)[:5]]).T
    aligned = np.array([np.asarray(aligned_x)[:5], np.asarray(aligned_y)[:5]]).T
    assert np.allclose(convert_to_hkl(dataset.sample, points),
                       convert_to_hkl(align_to_lattice(dataset.sample), aligned))
