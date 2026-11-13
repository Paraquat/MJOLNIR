import pytest

import warnings
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data.CameaDataReader import CameaDataReader
from MJOLNIR.Geometry import Instrument

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

@pytest.fixture(scope='module')
def sample():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return CameaDataReader(files[0]).sample

def test_converters():
    """A3, A4 of a point and back"""
    for qx, qy, Ei, Ef in [(1.0, 0.5, 5.0, 4.0), (-0.7, 1.2, 8.0, 3.5)]:
        A3, A4 = Instrument.converterToA3A4(qx, qy, Ei, Ef)
        assert np.allclose(np.ravel(Instrument.converterToQxQy(A3, A4, Ei, Ef)), [qx, qy])

@needs_data
@pytest.mark.parametrize('instrument, axes', [('CAMEA', 9), ('MultiFLEXX', 6), ('Bambus', 6)])
def test_prediction(sample, instrument, axes):
    assert instrument in Instrument.predictionInstrumentSupport
    clicks = []
    Ax = Instrument.prediction(A3Start=-40, A3Stop=40, A3Steps=41, A4Positions=[-40, -44], Ei=5.0,
                               sample=sample, instrument=instrument, outputFunction=clicks.append)
    assert len(Ax) == axes
    assert all(len(ax.lines) > 0 for ax in Ax) # The coverage of each analyser, all on the last
    assert len(Ax[-1].lines) == sum(len(ax.lines) for ax in Ax[:-1])
    assert Ax[0].get_xlabel().endswith('RLU') and Ax[-1].get_title() == 'All Planes'
    assert 'A3 = ' in Ax[0].format_coord(0.5, 0.5) and 'Ef = ' in Ax[0].format_coord(0.5, 0.5)
    Ax[0].get_figure().canvas.draw() # The lattice grids draw

@needs_data
def test_prediction_with_sample_environment(sample):
    Ax = Instrument.prediction(A3Start=-40, A3Stop=40, A3Steps=41, A4Positions=[-40], Ei=5.0,
                               sample=sample, PillarSizes=[20.0], PillarOffsets=[30.0],
                               SEAlignmentPeak=[1, 0, 0])
    plain = Instrument.prediction(A3Start=-40, A3Stop=40, A3Steps=41, A4Positions=[-40], Ei=5.0,
                                  sample=sample)
    assert len(Ax[0].lines) == len(plain[0].lines) + 2 # The pillar and its shadow
    with pytest.raises(AttributeError, match='does not match'):
        Instrument.prediction(A3Start=-40, A3Stop=40, A3Steps=41, A4Positions=[-40], Ei=5.0,
                              sample=sample, PillarSizes=[20.0], PillarOffsets=[30.0, 60.0],
                              SEAlignmentPeak=[1, 0, 0])
    with pytest.raises(AttributeError, match='not understood'):
        Instrument.prediction(-40, 40, 41, [-40], 5.0, sample, instrument='Tas')
