import pytest
import numpy as np

from MJOLNIR.Data.AxisView import AxisView

def test_axisview():
    arr1 = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]])
    arr2 = np.array([[10.0, 11.0, 12.0],
                     [13.0, 14.0, 15.0],
                     [16.0, 17.0, 18.0]])
    arr3 = np.array([[19.0, 20.0, 21.0],
                     [22.0, 23.0, 24.0],
                     [25.0, 26.0, 27.0]])

    # View along axis 0, i.e. arrays concatenated along axis 0
    view1 = AxisView([arr1, arr2, arr3], axis = 0)
    assert len(view1) == 9
    assert view1[0, 0] == pytest.approx(1.0)
    assert view1[1, 0] == pytest.approx(4.0)
    assert view1[3, 0] == pytest.approx(10.0)
    assert view1[8, 0] == pytest.approx(25.0)

    # slicing
    slice1 = view1[2:5, 0]
    assert len(slice1) == 3
    assert slice1[0] == pytest.approx(7.0)
    assert slice1[1] == pytest.approx(10.0)
    assert slice1[2] == pytest.approx(13.0)

    # iteration
    count = 0;
    for val in view1[:, 0]:
        if count == 0:
            assert val == pytest.approx(1.0)
        if count == 1:
            assert val == pytest.approx(4.0)
        if count == 3:
            assert val == pytest.approx(10.0)
        if count == 8:
            assert val == pytest.approx(25.0)
        count += 1

    # view along axis 1
    view2 = AxisView([arr1, arr2, arr3], axis = 1)
    assert len(view2) == 9
    assert view2[0, 0] == pytest.approx(1.0)
    assert view2[1, 0] == pytest.approx(2.0)
    assert view2[3, 0] == pytest.approx(10.0)
    assert view2[8, 0] == pytest.approx(21.0)
    slice2 = view2[2:5, 0]
    assert len(slice2) == 3
    assert slice2[0] == pytest.approx(3.0)
    assert slice2[1] == pytest.approx(10.0)
    assert slice2[2] == pytest.approx(11.0)

# Arrays of different lengths along the view axis, as for scans with different numbers of steps
@pytest.fixture
def uneven():
    first = np.arange(6, dtype=np.uint32).reshape(2, 3)     # rows 0, 1 of the view
    second = np.arange(6, 15, dtype=np.uint32).reshape(3, 3) # rows 2, 3, 4 of the view
    return AxisView([first, second], axis = 0), first, second

def test_uneven_lengths(uneven):
    view, first, second = uneven
    assert len(view) == 5
    assert view.shape == (5, 3)
    assert view[1, 2] == first[1, 2]
    assert view[2, 1] == second[0, 1] # First row of the second array
    assert view[4, 0] == second[2, 0]

def test_negative_index(uneven):
    view, first, second = uneven
    assert view[-1, 0] == second[-1, 0]
    assert view[-3, 1] == second[0, 1]
    assert view[-5, 2] == first[0, 2]

def test_slices(uneven):
    view, first, second = uneven
    assert np.array_equal(view[1:4, 2], [first[1, 2], second[0, 2], second[1, 2]])
    assert np.array_equal(view[:, 0], np.concatenate([first[:, 0], second[:, 0]]))
    assert np.array_equal(view[::-1, 0], np.concatenate([first[:, 0], second[:, 0]])[::-1])
    assert np.array_equal(view[::2, 1], [first[0, 1], second[0, 1], second[2, 1]])
    assert len(view[3:3, 0]) == 0

def test_values_are_float64(uneven):
    view, first, second = uneven
    assert first.dtype == np.uint32
    assert isinstance(view[0, 0], np.float64)
    assert view[1:3, 0].dtype == np.float64

def test_index_errors(uneven):
    view, first, second = uneven
    with pytest.raises(IndexError):
        view[5, 0]
    with pytest.raises(IndexError):
        view[-6, 0]
    with pytest.raises(IndexError): # Too many indices
        view[1, 0, 0]
    with pytest.raises(IndexError): # Too few indices
        view[1,]

def test_index_types(uneven):
    view, first, second = uneven
    with pytest.raises(TypeError): # Must be a tuple, even for one index
        view[1]
    with pytest.raises(TypeError): # Only the view axis can be sliced
        view[1, 0:2]
    with pytest.raises(TypeError): # No index arrays
        view[[0, 1], 0]

def test_numpy_integer_index(uneven):
    # Indices computed with numpy (np.argmax, np.arange) are numpy integers
    view, first, second = uneven
    assert view[np.int64(2), np.int64(1)] == second[0, 1]
    assert view[np.int32(-1), 0] == second[-1, 0]
    assert np.array_equal(view[1:3, np.uint8(2)], [first[1, 2], second[0, 2]])
    assert [view[i, 0] for i in np.arange(len(view))] == list(view[:, 0])

def test_iteration(uneven):
    view, first, second = uneven
    rows = list(view)
    assert len(rows) == 5
    assert np.array_equal(np.array(rows), np.concatenate([first, second]))

def test_iteration_is_float64(uneven):
    # Iteration converts to float64, like indexing
    view, first, second = uneven
    assert all(row.dtype == np.float64 for row in view)
    one_dimensional = AxisView([np.array([1, 2], dtype=np.uint32), np.array([3], dtype=np.uint32)])
    values = list(one_dimensional)
    assert values == [1.0, 2.0, 3.0]
    assert all(isinstance(value, np.float64) for value in values)

def test_three_dimensions():
    first = np.arange(24.0).reshape(2, 3, 4)
    second = np.arange(24.0, 48.0).reshape(2, 3, 4)

    view = AxisView([first, second], axis = 0)
    assert view.shape == (4, 3, 4)
    assert view[2, 1, 3] == second[0, 1, 3]
    assert np.array_equal(view[1:3, 2, 3], [first[1, 2, 3], second[0, 2, 3]])

    # A middle axis: the other axes keep their order
    view = AxisView([first, second], axis = 1)
    assert view.shape == (6, 2, 4)
    assert view[3, 1, 2] == second[1, 0, 2]
    assert np.array_equal(view[2:4, 0, 1], [first[0, 2, 1], second[0, 0, 1]])
    assert next(iter(view)).shape == (2, 4)

def test_negative_axis():
    first = np.arange(6.0).reshape(2, 3)
    second = np.arange(6.0, 12.0).reshape(2, 3)
    view = AxisView([first, second], axis = -1)
    assert view.shape == (6, 2)
    assert view[3, 0] == second[0, 0]
    with pytest.raises(np.exceptions.AxisError):
        AxisView([first, second], axis = 2)

def test_no_arrays():
    with pytest.raises(ValueError):
        AxisView([])

def test_mismatched_shapes():
    with pytest.raises(ValueError): # Other axes must match
        AxisView([np.zeros((2, 3)), np.zeros((2, 4))], axis = 0)
    # Only the view axis may differ
    assert AxisView([np.zeros((2, 3)), np.zeros((2, 4))], axis = 1).shape == (7, 2)
