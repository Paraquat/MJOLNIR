import pytest

import numpy as np
from pathlib import Path

from MJOLNIR import _tools
from MJOLNIR.Data.AbsoluteNormalization import absolute_normalization_factor, MAGNETIC_CONSTANT
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))
needs_data = pytest.mark.skipif(not files, reason='sampledata/camea2024*.hdf not available')
HKL1, HKL2 = [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]

def test_factor():
    molar_mass = _tools.calculateMolarMass('TbFeO3')
    assert absolute_normalization_factor(2.0, 'TbFeO3') == pytest.approx(
        4*np.pi*2.0*2.0**2/(molar_mass*MAGNETIC_CONSTANT))
    assert absolute_normalization_factor(2.0, sample_molar_mass=100.0, sample_g_factor=1.0) \
        == pytest.approx(4*np.pi*2.0/(100.0*MAGNETIC_CONSTANT))
    assert absolute_normalization_factor(2.0, sample_molar_mass=100.0, units='nuclear') \
        == pytest.approx(4*np.pi*2.0/100.0) # No g^2, no 13.77
    vanadium = _tools.calculateMolarMass('V')/(2.0*15.25*5.08*100000)
    assert absolute_normalization_factor(2.0, sample_molar_mass=100.0, units='nuclear',
                                         correct_vanadium=True) \
        == pytest.approx(4*np.pi*2.0*vanadium/100.0)

@pytest.mark.parametrize('extra', [{}, dict(correctVanadium=True),
                                   dict(sampleGFactor=1.5, formulaUnitsPerUnitCell=4)])
def test_factor_matches_old(extra):
    names = {'correctVanadium': 'correct_vanadium', 'sampleGFactor': 'sample_g_factor',
             'formulaUnitsPerUnitCell': 'formula_units_per_unit_cell'}
    old = _tools.calculateAbsoluteNormalization(sampleMass=2.0, sampleChemicalFormula='TbFeO3',
                                               **extra)
    new = absolute_normalization_factor(2.0, 'TbFeO3',
                                        **{names[key]: value for key, value in extra.items()})
    assert new == pytest.approx(old, rel=1e-14)

def test_factor_needs_a_sample():
    with pytest.raises(ValueError, match='sample mass'):
        absolute_normalization_factor(None, 'TbFeO3')
    with pytest.raises(ValueError, match='sample mass'):
        absolute_normalization_factor(0.0, 'TbFeO3')
    with pytest.raises(ValueError, match='molar mass or chemical formula'):
        absolute_normalization_factor(2.0)
    with pytest.raises(ValueError, match="'magnetic' or 'nuclear'"):
        absolute_normalization_factor(2.0, 'TbFeO3', units='atomic')

@pytest.fixture
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

def cut(dataset, **options):
    return dataset.cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8, **options).data

@needs_data
def test_normalize_and_undo(dataset):
    before = cut(dataset)
    normalization = [reader.normalization.copy() for reader in dataset._readers]
    factor = dataset.absolute_normalize(2.0, 'TbFeO3')
    assert dataset.absolute_normalization == factor
    after = cut(dataset)
    assert np.allclose(after['Int'], before['Int']/factor, equal_nan=True)
    assert np.allclose(after['Int_err'], before['Int_err']/factor, equal_nan=True)
    assert np.array_equal(after['intensity'], before['intensity']) # Counts untouched
    # A second call replaces the first
    second = dataset.absolute_normalize(1.0, 'TbFeO3')
    assert np.allclose(cut(dataset)['Int'], before['Int']/second, equal_nan=True)
    dataset.undo_absolute_normalize() # The old undo multiplied by the factor again
    assert dataset.absolute_normalization == 1.0
    assert all(np.array_equal(reader.normalization, original)
               for reader, original in zip(dataset._readers, normalization))

@needs_data
def test_kept_through_changes(dataset):
    factor = dataset.absolute_normalize(2.0, 'TbFeO3')
    normalised = cut(dataset)
    dataset.set_binning(3) # The old reconversion lost the normalisation
    dataset.set_binning(8)
    assert np.allclose(cut(dataset)['Int'], normalised['Int'], equal_nan=True)
    dataset.append(files[2])
    assert dataset._readers[-1].normalization_factor == factor
    subset = dataset[0:2]
    assert subset.absolute_normalization == factor
    assert all(reader.normalization_factor == factor for reader in subset._readers)
    difference = dataset - dataset
    assert difference.absolute_normalization == factor
    difference.undo_absolute_normalize() # Allowed on a difference: only the normalization
    assert all(reader.normalization_factor == 1.0 for reader in difference._readers)
    assert all(reader.normalization_factor == factor for reader in dataset._readers)

@needs_data
def test_with_background(dataset):
    dataset.generate_background_model()
    before = cut(dataset, background_subtraction=True)
    factor = dataset.absolute_normalize(2.0, 'TbFeO3')
    after = cut(dataset, background_subtraction=True)
    # Foreground and background both in the new units: the difference scales too
    for column in ['Int', 'Int_Fg', 'Int_Bg', 'Int_err']:
        assert np.allclose(after[column], before[column]/factor, equal_nan=True), column
