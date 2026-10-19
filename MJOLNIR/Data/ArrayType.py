from enum import StrEnum

class ArrayType(StrEnum):
    """Names of the arrays of a data reader: what a DataSet can view as one contiguous array,
    and what masks can use as coordinates. Values are the attribute names on the reader (dotted
    for nested attributes). Members are strings, so ArrayType.QX == 'qx'"""
    # Raw data
    COUNTS = "counts"
    A3 = "A3"
    A3OFF = "A3Offset"
    A4 = "A4"
    A4OFF = "A4Offset"
    TWOTHETA = "twotheta"
    EI = "Ei"
    MONITOR1 = "monitor1"
    MONITOR2 = "monitor2"

    # Instrument calibration
    EF_CAL = "calibration.Ef"
    WIDTH_CAL = "calibration.width"
    BG_CAL = "calibration.bg"
    AMP_CAL = "calibration.amp"
    A4_CAL = "calibration.A4"
    BOUND_CAL = "calibration.bound"

    # Derived data
    INTENSITY = "intensity"
    INTENSITY_ERROR = "intensity_error" # None unless set (see DataReader.intensity_error)
    NORMALIZATION = "normalization"
    MONITOR = "monitor"
    MASK = "mask"
    BACKGROUND = "background"

    # Sample environment, one value per scan step
    TEMPERATURE = "environment.temperature"
    TEMPERATURE_SETPOINT = "environment.temperature_setpoint"
    MAGNETIC_FIELD = "environment.magnetic_field"
    MAGNETIC_FIELD_SETPOINT = "environment.magnetic_field_setpoint"
    ELECTRIC_FIELD = "environment.electric_field"
    ELECTRIC_FIELD_SETPOINT = "environment.electric_field_setpoint"
    QX = "qx"
    QY = "qy"
    ENERGY = "energy"
    H = "h"
    K = "k"
    L = "l"
