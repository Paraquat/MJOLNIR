import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

from MJOLNIR.Data.DataWriter import DataWriter
from MJOLNIR.Data.NexusDataReader import NexusDataReader
from MJOLNIR.Data.NexusLayout import Field, NexusLayout

class NexusDataWriter(DataWriter):
    """Save a Nexus file by updating a copy of the file the reader read: everything in the
    original file is kept, and only the values given to set() (and what a subclass adds in
    update()) are written. The copy is completed under a temporary name and then moved into
    place, so a failed save leaves no file behind"""

    layout: type[NexusLayout] # Defaults to the reader's layout

    def __init__(self, reader: NexusDataReader):
        super().__init__(reader)
        if not hasattr(type(self), 'layout'):
            self.layout = reader.layout
        elif reader.layout is not self.layout:
            raise ValueError(f'{type(self).__name__} writes {self.layout.__name__} files, '
                             f'not {reader.layout.__name__} ({reader.file.name})')
        self.values: dict[str, np.ndarray] = {}

    def set(self, key: str, value):
        """Value to write at a layout field (KeyError if the layout has no such field)"""
        self.layout.field(key)
        self.values[key] = np.asarray(value)

    def save(self, path: Path, overwrite: bool = False) -> Path:
        path = self._check_target(path, overwrite)
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.',
                                             suffix='.tmp')
        os.close(handle)
        try:
            shutil.copyfile(self.reader.file, temporary)
            with h5py.File(temporary, 'r+') as f:
                self.update(f)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    def update(self, f: h5py.File):
        """Write the changes into the copy, f. Subclasses extend this"""
        for key, value in self.values.items():
            self.write_field(f, self.layout.field(key), value)

    def write_field(self, f: h5py.File, field: Field, value):
        """Write value at field.path with the field's dtype and units. A dataset of the same
        shape and dtype is overwritten in place (keeping its attributes and hard links, e.g.
        entry/data/* in NICOS files); otherwise it is replaced, keeping its attributes"""
        value = np.asarray(value)
        dtype = np.dtype(field.dtype) if field.dtype is not None else value.dtype
        existing = f.get(field.path)
        if isinstance(existing, h5py.Dataset) and existing.shape == value.shape \
                and existing.dtype == dtype:
            existing[...] = value
            dataset = existing
        else:
            # Attributes with their stored HDF5 types (fixed or variable length strings)
            attributes = [] if not isinstance(existing, h5py.Dataset) else \
                [(name, existing.attrs[name], existing.attrs.get_id(name).dtype)
                 for name in existing.attrs]
            if existing is not None:
                del f[field.path]
            self._create_parents(f, field.path)
            dataset = f.create_dataset(field.path, data=value.astype(dtype))
            for name, attribute, attribute_dtype in attributes:
                dataset.attrs.create(name, attribute, dtype=attribute_dtype)
        units = dataset.attrs.get('units')
        if isinstance(units, bytes):
            units = units.decode()
        if field.units is not None and units != field.units: # Else keep the stored attribute
            dataset.attrs['units'] = np.bytes_(field.units)

    def _create_parents(self, f: h5py.File, path: str):
        """Create the missing groups above path, with their NX_class from the layout"""
        classes = {group.path: group.nx_class for group in self.groups().values()}
        parent = PurePosixPath(path).parent
        for ancestor in reversed([parent, *parent.parents]):
            name = str(ancestor)
            if name in ('.', '/') or name in f:
                continue
            group = f.create_group(name)
            if classes.get(name) is not None:
                group.attrs['NX_class'] = np.bytes_(classes[name])

    def groups(self):
        """Groups whose NX_class the writer knows, filled in. Subclasses add e.g. calibration
        groups"""
        return self.layout.all_groups()
