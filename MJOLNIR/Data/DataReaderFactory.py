from pathlib import Path

import MJOLNIR.Data.DataReader as DataReader
from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Data.InstrumentName import InstrumentName

class DataReaderFactory:

    _registry: dict(InstrumentName, type[DataReader]) = {}

    @classmethod
    def register(cls, name: InstrumentName, reader: type[DataReader]) -> None:
        cls._registry[name] = reader

    @classmethod
    def create(cls, name: InstrumentName, path: Path) -> DataReader:
        try:
            reader = cls._registry[name]
        except KeyError:
            raise ValueError(f"Unknown DataReader type: {name.value}") from None
        return reader(path)

    def __init__(self):
        self.register(InstrumentName.CAMEA, CameaDataReader)
