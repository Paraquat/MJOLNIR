"""Absolute normalisation of intensities against the vanadium normalisation (the old
DataSet.absoluteNormalize and _tools.calculateAbsoluteNormalization)."""
import numpy as np

from MJOLNIR import _tools

# 1/(gamma r0 / 2)^2 [1/barn]: (gamma r0 / 2)^2 = 0.07265 barn turns a cross-section in barn
# into magnetic units (muB^2) together with g^2
MAGNETIC_CONSTANT = 13.77

def _molar_mass(molar_mass, chemical_formula, units_per_unit_cell, name: str) -> float:
    if molar_mass is not None:
        return float(molar_mass)
    if chemical_formula is None:
        raise ValueError(f'Absolute normalisation needs the {name} molar mass or chemical formula')
    return _tools.calculateMolarMass(sampleChemicalFormula=chemical_formula,
                                     formulaUnitsPerUnitCell=units_per_unit_cell)

def absolute_normalization_factor(sample_mass: float, sample_chemical_formula: str | None = None,
                                  sample_molar_mass: float | None = None,
                                  formula_units_per_unit_cell: float = 1.0,
                                  sample_g_factor: float = 2.0, units: str = 'magnetic',
                                  correct_vanadium: bool = False,
                                  vanadium_mass: float = 15.25,
                                  vanadium_chemical_formula: str = 'V',
                                  vanadium_molar_mass: float | None = None,
                                  vanadium_units_per_unit_cell: float = 1.0,
                                  vanadium_monitor: float = 100000,
                                  vanadium_sigma_incoherent: float = 5.08,
                                  vanadium_g_factor: float = 2.0) -> float:
    """Factor the normalisation is multiplied by (intensities divided by) to put them in
    absolute units, relative to the vanadium the normalisation tables were measured on.

        magnetic units (muB^2 per formula unit): 4 pi m g^2 V / (M 13.77)
        nuclear units (barn per formula unit):   4 pi m V / M

    m is the sample mass [g], M its molar mass per formula unit [g/mol] (from the chemical
    formula if not given), g its g-factor and 13.77 = 1/(gamma r0 / 2)^2 [1/barn]. V is 1, or
    with correct_vanadium (normalisation tables not yet scaled to the vanadium sample)
    M_V / (g_V m_V sigma_inc,V monitor_V), as the old calculateAbsoluteNormalization.

    The sample mass and its molar mass or chemical formula are required: without them the
    result cannot be absolute (the old code then applied the unit conversion alone, a factor
    16 pi/13.77 = 3.65).
    """
    if sample_mass is None or not sample_mass > 0:
        raise ValueError(f'Absolute normalisation needs the sample mass in g (> 0); got {sample_mass}')
    molar_mass = _molar_mass(sample_molar_mass, sample_chemical_formula,
                             formula_units_per_unit_cell, 'sample')
    if correct_vanadium:
        vanadium_molar = _molar_mass(vanadium_molar_mass, vanadium_chemical_formula,
                                     vanadium_units_per_unit_cell, 'vanadium')
        vanadium = vanadium_molar/(vanadium_g_factor*vanadium_mass*vanadium_sigma_incoherent
                                   *vanadium_monitor)
    else:
        vanadium = 1.0
    if units == 'magnetic':
        return 4*np.pi*sample_mass*sample_g_factor**2*vanadium/(molar_mass*MAGNETIC_CONSTANT)
    if units == 'nuclear':
        return 4*np.pi*sample_mass*vanadium/molar_mass
    raise ValueError(f"units must be 'magnetic' or 'nuclear'; got {units!r}")
