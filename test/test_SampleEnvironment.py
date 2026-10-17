import pytest

import numpy as np
from pathlib import Path

from MJOLNIR.Data import Mask
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.FlatconeDataReader import FlatconeDataReader
from MJOLNIR.Data.MarmotDataReader import MarmotDataReader
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

camea_files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
camea_2023_file = Path.home() / 'projects' / 'camea_test' / 'camea2023n002630.hdf'
flatcone_file = Path.home() / 'projects' / 'marmot+flatcone' / '043555_pastis.nxs'
marmot_file = Path.home() / 'projects' / 'marmot+flatcone' / '054724.nxs'

def test_camea():
    environment = CameaDataReader(camea_files[0]).environment
    assert environment.temperature.shape == (135,)
    assert environment.temperature[0] == pytest.approx(3.2084, abs=1e-4)
    assert np.allclose(environment.magnetic_field, 1.5014, atol=1e-4)
    assert environment.units == {'temperature': 'K', 'magnetic_field': 'T'}
    # Not recorded by CAMEA
    assert environment.temperature_setpoint is None
    assert environment.magnetic_field_setpoint is None
    assert environment.electric_field is None

def test_camea_older_names():
    # NICOS files of 2023 store the temperature as entry/sample/T
    environment = CameaDataReader(camea_2023_file).environment
    assert environment.temperature.shape == (111,)
    assert environment.temperature[0] == pytest.approx(5.0044, abs=1e-4)
    assert environment.magnetic_field[0] == pytest.approx(-0.0002, abs=1e-6)

def test_setpoints():
    # ILL files record setpoints; single values are repeated for every scan step
    flatcone = FlatconeDataReader(flatcone_file).environment
    assert flatcone.temperature.shape == flatcone.temperature_setpoint.shape == (6,)
    assert np.all(flatcone.temperature_setpoint == 1.5)
    marmot = MarmotDataReader(marmot_file).environment
    assert marmot.temperature.shape == (361,)
    assert np.all(marmot.temperature == marmot.temperature[0]) # One value in the file
    assert np.all(marmot.temperature_setpoint == 2.0)

def test_per_step():
    reader = CameaDataReader(camea_files[0])
    assert np.array_equal(reader.per_step(4.2, 'x'), np.full(135, 4.2))
    assert reader.per_step(np.arange(135), 'x').shape == (135,)
    with pytest.warns(UserWarning, match='134 values of x for 135 scan steps'):
        assert reader.per_step(np.arange(134), 'x') is None

def test_dataset_views():
    dataset = DataSet(InstrumentName.CAMEA, files=camea_files[:2])
    temperature = dataset.get_view(ArrayType.TEMPERATURE)
    assert len(temperature) == 270 # One value per scan step
    assert temperature[135,] == pytest.approx(dataset[1].environment.temperature[0])
    field = np.mean(list(dataset.get_view(ArrayType.MAGNETIC_FIELD)))
    assert field == pytest.approx(1.5014, abs=1e-3)
    with pytest.raises(AttributeError): # Not recorded
        dataset.get_view(ArrayType.TEMPERATURE_SETPOINT)

def test_mask_by_temperature():
    dataset = DataSet(InstrumentName.CAMEA, files=camea_files[:1])
    dataset.use_default_mask = False
    temperature = dataset[0].environment.temperature
    limit = np.median(temperature)
    dataset.mask = Mask.lineMask(start=0, end=limit, coordinates=ArrayType.TEMPERATURE)
    mask = dataset.mask[0]
    assert mask.shape == (135, 104, 64) # Spread over each step's data
    assert np.array_equal(mask[:, 0, 0], temperature <= limit)
