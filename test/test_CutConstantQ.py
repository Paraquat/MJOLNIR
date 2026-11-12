import pytest

import warnings
import numpy as np
from pathlib import Path

from MJOLNIR import _tools
from MJOLNIR.Data import Cuts
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

SAMPLE_DATA = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not SAMPLE_DATA, reason='sampledata/camea2024*.hdf not available')

def toy_points(**extra):
    """Points at distances 0, 0.03, 0.049, 0.051 from q = (1, 0), energies 1, 1, 2, 2"""
    qx = np.array([1.0, 1.03, 1.0, 1.051])
    qy = np.array([0.0, 0.0, 0.049, 0.0])
    points = dict(qx=qx, qy=qy, energy=np.array([1.0, 1.0, 2.0, 2.0]),
                  intensity=np.array([4.0, 9.0, 16.0, 25.0]),
                  normalization=np.full(4, 2.0), monitor=np.full(4, 10.0))
    points.update(extra)
    return points

def test_width_is_the_diameter():
    result = Cuts.cut_constant_q(**toy_points(), q=[1.0, 0.0], width=0.1, min_pixel=0.5,
                                 energy_min=0.0, energy_max=3.0)
    data = result.data
    assert data['BinCount'].sum() == 3 # 0.051 is outside width/2
    assert result.bins['energy'] == pytest.approx([0.95, 1.5, 2.0]) # _tools.binEdges
    assert list(data['Energy']) == pytest.approx([1.225, 1.75]) # Bin centres
    assert list(data['intensity']) == [13.0, 16.0]
    assert list(data['Int']) == pytest.approx([13*2/(4*20), 16/(2*10)])
    assert list(data['Int_err']) == pytest.approx([np.sqrt(13)*2/(4*20), 4/(2*10)])
    assert (data['Qx'] == 1.0).all() and (data['Qy'] == 0.0).all()
    assert data['H'].isna().all() # No sample
    assert result.parameters['width'] == 0.1

def test_energy_limits_and_constant_bins():
    points = toy_points()
    inclusive = Cuts.cut_constant_q(**points, q=[1.0, 0.0], width=0.1, min_pixel=0.5,
                                    energy_min=1.0, energy_max=2.0)
    assert inclusive.data['BinCount'].sum() == 3 # Both limits included
    constant = Cuts.cut_constant_q(**points, q=[1.0, 0.0], width=0.1, min_pixel=0.25,
                                   energy_min=0.0, energy_max=3.0, constant_bins=True)
    assert np.allclose(constant.bins['energy'], [1.0, 1.25, 1.5, 1.75, 2.0])
    assert constant.data['BinCount'].sum() == 3

def test_measurement_errors():
    sigma = np.array([1.0, 2.0, 3.0, 4.0])
    data = Cuts.cut_constant_q(**toy_points(intensity_error=sigma), q=[1.0, 0.0], width=0.1,
                               min_pixel=0.5, energy_min=0.0, energy_max=3.0).data
    # Variances summed per bin, then the square root, normalised like Int
    assert list(data['Int_err']) == pytest.approx([np.sqrt(1 + 4)*2/(4*20), 3/(2*10)])
    assert list(data['Int']) == pytest.approx([13*2/(4*20), 16/(2*10)]) # Unchanged

def test_background_with_errors():
    sigma = np.array([1.0, 2.0, 3.0, 4.0])
    background = np.array([1.0, 1.0, 2.0, 2.0])
    data = Cuts.cut_constant_q(**toy_points(intensity_error=sigma, background=background),
                               q=[1.0, 0.0], width=0.1, min_pixel=0.5, energy_min=0.0,
                               energy_max=3.0).data
    scale = np.array([2/(4*20), 1/(2*10)])
    assert list(data['intensity']) == [13.0, 16.0] # The measured counts, as in cut_1d
    assert list(data['intensity_Bg']) == [2.0, 2.0]
    assert list(data['Int']) == pytest.approx((np.array([13, 16]) - 2)*scale)
    assert list(data['Int_Fg_err']) == pytest.approx(np.sqrt([5, 9])*scale)
    assert list(data['Int_err']) == pytest.approx(np.sqrt(np.array([5, 9]) + 2)*scale)

def test_empty_and_wrong_q():
    result = Cuts.cut_constant_q(**toy_points(), q=[5.0, 5.0], width=0.1, min_pixel=0.5,
                                 energy_min=0.0, energy_max=3.0)
    assert len(result) == 0 and 'Int_err' in result.data
    with pytest.raises(ValueError, match='Qx, Qy'):
        Cuts.cut_constant_q(**toy_points(), q=[1.0, 0.0, 0.0], width=0.1, min_pixel=0.5,
                            energy_min=0.0, energy_max=3.0)

def data_point(dataset):
    """H, K, L of a measured point, and an energy range inside the data"""
    qx, qy = np.asarray(dataset.gather(ArrayType.QX)), np.asarray(dataset.gather(ArrayType.QY))
    energy = np.asarray(dataset.gather(ArrayType.ENERGY))
    i = len(qx)//3
    return (dataset.convert_to_hkl(np.array([qx[i], qy[i]])),
            float(np.percentile(energy, 10)), float(np.percentile(energy, 90)))

@needs_data
def test_files_aligned_with_rlu():
    dataset = DataSet(InstrumentName.CAMEA, files=SAMPLE_DATA[:2])
    hkl, energy_min, energy_max = data_point(dataset)
    second = dataset.samples[1]
    second.RotMat = _tools.Rot(second.theta + 10) # Another orientation, as in test_gather_q
    result = dataset.cut_constant_q(hkl, width=0.1, min_pixel=0.05, energy_min=energy_min,
                                    energy_max=energy_max)
    qx, qy = dataset.gather_q() # The second file rotated into the first file's frame
    points = dict(qx=np.asarray(qx), qy=np.asarray(qy),
                  **{key: np.asarray(dataset.gather(array_type)) for key, array_type in
                     [('energy', ArrayType.ENERGY), ('intensity', ArrayType.INTENSITY),
                      ('normalization', ArrayType.NORMALIZATION),
                      ('monitor', ArrayType.MONITOR)]})
    expected = Cuts.cut_constant_q(**points, q=dataset.convert_to_qxqy(hkl), width=0.1,
                                   min_pixel=0.05, energy_min=energy_min,
                                   energy_max=energy_max, sample=dataset.sample)
    assert np.array_equal(result.data['Int'].to_numpy(), expected.data['Int'].to_numpy())
    unaligned = dataset.cut_constant_q(dataset.convert_to_qxqy(hkl), width=0.1, min_pixel=0.05,
                                       energy_min=energy_min, energy_max=energy_max, rlu=False)
    assert result.data['BinCount'].sum() != unaligned.data['BinCount'].sum()

@needs_data
def test_difference_errors():
    dataset = DataSet(InstrumentName.CAMEA, files=SAMPLE_DATA[:1])
    hkl, energy_min, energy_max = data_point(dataset)
    difference = dataset - dataset # Zero counts, variance 2 * counts
    reader = difference._readers[0]
    assert np.allclose(reader.intensity_error**2, 2*dataset._readers[0].intensity)
    original = dataset.cut_constant_q(hkl, width=0.1, min_pixel=0.05, energy_min=energy_min,
                                      energy_max=energy_max).data
    subtracted = difference.cut_constant_q(hkl, width=0.1, min_pixel=0.05,
                                           energy_min=energy_min, energy_max=energy_max).data
    assert np.allclose(subtracted['Int'], 0)
    assert np.allclose(subtracted['Int_err'], np.sqrt(2)*original['Int_err'])
