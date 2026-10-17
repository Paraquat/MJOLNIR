from pathlib import Path

import MJOLNIR.Data.DataReader as DataReader
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.BambusDataReader import BAMBUSDataReader
from MJOLNIR.Data.MultiFlexxDataReader import MultiFLEXXDataReader
from MJOLNIR.Data.FlatconeDataReader import FlatconeDataReader
from MJOLNIR.Data.MarmotDataReader import MarmotDataReader
from MJOLNIR.Data.InstrumentName import InstrumentName

class DataReaderFactory:

    _registry: dict[InstrumentName, type[DataReader]] = {}

    @classmethod
    def register(cls, name: InstrumentName, reader: type[DataReader]) -> None:
        cls._registry[name] = reader

    @classmethod
    def get_reader_type(cls, name: InstrumentName) -> type[DataReader]:
        try:
            return cls._registry[name]
        except KeyError:
            raise ValueError(f"Unknown DataReader type: {name.value}") from None

    @classmethod
    def get_instrument(cls, reader: type[DataReader]) -> InstrumentName | None:
        """Instrument a reader type is registered for, or None if it is not registered"""
        for name, registered in cls._registry.items():
            if registered is reader:
                return name
        return None

    @classmethod
    def create(cls, name: InstrumentName, path: Path) -> DataReader:
        return cls.get_reader_type(name)(path)

    def __init__(self):
        self.register(InstrumentName.CAMEA, CameaDataReader)
        self.register(InstrumentName.BAMBUS, BAMBUSDataReader)
        self.register(InstrumentName.MULTIFLEXX, MultiFLEXXDataReader)
        self.register(InstrumentName.FLATCONE, FlatconeDataReader)
        self.register(InstrumentName.MARMOT, MarmotDataReader)
