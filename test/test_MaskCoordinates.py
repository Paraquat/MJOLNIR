import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data import Mask
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

data_path = Path(__file__).parent.parent / 'sampledata'
files = sorted(data_path.glob('camea2024*.hdf'))

@pytest.fixture(scope="module")
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

def test_array_type_is_string():
    assert ArrayType.QX == 'qx'
    assert isinstance(ArrayType.QX, str)
    assert ArrayType('qx') is ArrayType.QX
    assert str(ArrayType.EF_CAL) == 'calibration.Ef'
    with pytest.raises(ValueError):
        ArrayType('Qx')

def test_view_by_name(dataset):
    assert dataset.get_view('qx').shape == dataset.get_view(ArrayType.QX).shape == (270, 104, 64)
    with pytest.raises(ValueError):
        dataset.get_view('not_an_array')

def test_enum_and_string_coordinates_agree(dataset):
    pairs = [
        (Mask.circleMask(center=[-1.0, 0.0], radius=0.2, coordinates=[ArrayType.QX, ArrayType.QY]),
         Mask.circleMask(center=[-1.0, 0.0], radius=0.2, coordinates=['qx', 'qy'])),
        (Mask.lineMask(start=0.0, end=0.5, coordinates=ArrayType.ENERGY),
         Mask.lineMask(start=0.0, end=0.5, coordinates='energy')),
        (Mask.rectangleMask([-1, 0], [0, 1], coordinates=[ArrayType.H, ArrayType.K]),
         Mask.rectangleMask([-1, 0], [0, 1], coordinates=['h', 'k'])),
    ]
    for by_enum, by_string in pairs:
        assert by_enum == by_string
        masked = [m.sum() for m in by_enum(dataset)]
        assert masked == [m.sum() for m in by_string(dataset)]
        assert sum(masked) > 0

def test_unknown_coordinate_on_data_file(dataset):
    mask = Mask.circleMask(center=[0, 0], radius=0.1, coordinates=['Qx', 'Qy'])
    with pytest.raises(AttributeError, match="'Qx' is not an array of a data file. Use one of: .*qx"):
        mask(dataset)

def test_any_attribute_on_other_objects():
    class Points:
        X = np.array([0.0, 1.0, 2.0])
        Y = np.array([0.0, 0.0, 0.0])
    mask = Mask.circleMask(center=[0, 0], radius=1.5, coordinates=['X', 'Y'])
    assert list(mask(Points())) == [True, True, False]

def test_coordinates_saved_as_strings(tmp_path):
    mask = Mask.circleMask(center=[0, 0], radius=1, coordinates=[ArrayType.QX, ArrayType.QY])
    assert all(type(coordinate) is str for coordinate in mask.coordinates)
    mask.save(str(tmp_path / 'circle.mask'))
    loaded = Mask.load(str(tmp_path / 'circle.mask'))
    assert loaded == mask
    assert loaded.coordinates == ['qx', 'qy']
