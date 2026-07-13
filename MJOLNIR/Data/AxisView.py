from bisect import bisect_right
from collections.abc import Iterator, Sequence
from typing import overload

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class AxisView(Sequence[FloatArray]):
    """Take a container of numpy float arrays and return a view that is
    concatenated along the given axis. Use bisection to find elements faster"""

    def __init__(self, arrays: list[FloatArray], axis: int = 0) -> None:
        self._arrays = arrays
        self._axis = np._core.numeric.normalize_axis_index(axis, arrays[0].ndim)

        self._offsets: list[int] = []
        self._length = 0

        for array in arrays:
            normalized_axis = np._core.numeric.normalize_axis_index(axis, array.ndim)
            if normalized_axis != self._axis:
                raise ValueError("Inconsistent array dimensions")
            self._offsets.append(self._length)
            self._length += array.shape[self._axis]

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[FloatArray]:
        for array in self._arrays:
            yield from np.moveaxis(array, self._axis, 0)

    def _locate(self, global_index: int) -> tuple[FloatArray, int]:
        if global_index < 0:
            global_index += len(self)

        if global_index < 0 or global_index >= len(self):
            raise IndexError("AxisView index out of range")

        array_index = bisect_right(self._offsets, global_index,) - 1
        local_index = global_index - self._offsets[array_index]

        return self._arrays[array_index], local_index

    def _make_array_index(self, array: FloatArray, local_index: int,
        fixed_indices: tuple[int, ...],) -> tuple[int, ...]:
        if len(fixed_indices) != array.ndim - 1:
            raise IndexError(
                f"Expected {array.ndim - 1} fixed indices, "
                f"received {len(fixed_indices)}"
            )

        result: list[int] = []
        fixed_iterator = iter(fixed_indices)

        for dimension in range(array.ndim):
            if dimension == self._axis:
                result.append(local_index)
            else:
                result.append(next(fixed_iterator))

        return tuple(result)

    @overload
    def __getitem__(self, key: tuple[int, int],) -> np.float64:
        ...

    @overload
    def __getitem__(self, key: tuple[slice, int],) -> FloatArray:
        ...

    def __getitem__(self, key: tuple[int | slice, ...],) -> np.float64 | FloatArray:
        if not isinstance(key, tuple):
            raise TypeError("Use multidimensional indexing, for example view[:, 0]")

        logical_index = key[0]
        fixed_indices = key[1:]

        if not all(isinstance(index, int) for index in fixed_indices):
            raise TypeError("Indices for non-view axes must be integers")

        fixed_indices = tuple(fixed_indices)

        if isinstance(logical_index, int):
            array, local_index = self._locate(logical_index)
            array_index = self._make_array_index(array, local_index, fixed_indices,)
            return np.float64(array[array_index])

        if isinstance(logical_index, slice):
            indices = range(*logical_index.indices(len(self)))
            return np.asarray(
                [self[(index, *fixed_indices)] for index in indices], dtype=np.float64,)

        raise TypeError(
            "The first index must be an integer or slice")
