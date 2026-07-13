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
