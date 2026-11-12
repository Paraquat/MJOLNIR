"""The new code against the old code's saved output (test/reference, see reference_data.py and
reference/generate.py). Differences that are fixes of the old code are left out of the
comparisons and named where they are."""
import pytest

import warnings
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import reference_data as R
from MJOLNIR.Data import Mask, Plotting
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.DataReaderFactory import DataReaderFactory
from MJOLNIR.Data.DataSet import DataSet
from MJOLNIR.Data.InstrumentName import InstrumentName

HKL1, HKL2 = [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
QE_Q1, QE_Q2 = np.array([-0.5, 1.5, 0.0]), np.array([1.5, -0.5, 0.0])
QE_LINE = np.array([[-1, 0, 0], [0, 1, 0], [0.5, 0.5, 0], [0, 0, 0]], dtype=float)
QE_LINE_ENERGY = np.arange(-1.0, 0.81, 0.1)
POWDER_Q, POWDER_E = np.arange(0.4, 2.0, 0.05), np.arange(-1, 0.81, 0.1)
RAW_SELECTIONS = [(None, None), (0, 0), (50, 3), (103, 7)]
BIN3D_STEPS = [(0.05, 0.05, 0.1), (0.08, 0.03, 0.25)]

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')

def dataset(files) -> DataSet:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return DataSet(InstrumentName.CAMEA, files=files)

@pytest.fixture(scope='module')
def camea_2024():
    return dataset(R.CAMEA_2024[:2])

@pytest.fixture(scope='module')
def camea_2023():
    return dataset(R.CAMEA_2023)

# Conversion

def reader_for(case: str, folder: Path):
    factory = DataReaderFactory()
    if case.startswith('camea2024'):
        return factory.create(InstrumentName.CAMEA, R.CAMEA_2024[0])
    if case == 'camea2023_b8':
        return factory.create(InstrumentName.CAMEA, R.CAMEA_2023[0])
    if case == 'bambus':
        return factory.create(InstrumentName.BAMBUS, R.BAMBUS)
    if case == 'multiflexx':
        return factory.create(InstrumentName.MULTIFLEXX,
                              R.multiflexx_file(folder / 'multiflexx_synthetic'))
    if case == 'flatcone':
        return factory.create(InstrumentName.FLATCONE, R.FLATCONE)
    from MJOLNIR.Data.MarmotDataReader import MarmotDataReader
    return MarmotDataReader(R.MARMOT, calibration_file=R.marmot_table(folder / 'marmot.table'))

CONVERSION = {'camea2024_b8': (8, None, R.CAMEA_2024), 'camea2024_b3_a3offset': (3, 1.5, R.CAMEA_2024),
              'camea2023_b8': (8, None, R.CAMEA_2023), 'bambus': (None, None, R.BAMBUS),
              'multiflexx': (None, None, R.BAMBUS), 'flatcone': (None, None, R.FLATCONE),
              'marmot': (None, None, R.MARMOT)}

@pytest.mark.parametrize('case', CONVERSION)
def test_conversion(case, tmp_path):
    binning, offset, inputs = CONVERSION[case]
    if not (inputs if isinstance(inputs, list) else inputs.exists()):
        pytest.skip(f'input data not available: {inputs}')
    reference = R.load('conversion')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reader = reader_for(case, tmp_path)
        if offset is not None:
            reader.A3Offset = np.array([offset])
        reader.set_instrument_calibration(binning)
        reader.transform()
    for old, new in [('I', 'intensity'), ('Norm', 'normalization'), ('Monitor', 'monitor'),
                     ('qx', 'qx'), ('qy', 'qy'), ('energy', 'energy'), ('h', 'h'), ('k', 'k'),
                     ('l', 'l')]:
        values, rtol, atol = getattr(reader, new), 1e-12, 0.0
        if case == 'flatcone' and old == 'I':
            # Flatcone counts are floats; the old conversion truncated them toward zero
            values = np.trunc(values)
        if case == 'marmot' and old in 'hkl':
            # The old Marmot loader kept the unit cell in single precision (UB differs ~1e-7)
            rtol, atol = 1e-6, 1e-6
        R.assert_summary(reference, f'{case}/{old}', values, rtol=rtol, atol=atol)
    for name in ['A3', 'A4', 'Ei']:
        assert np.allclose(np.ravel(getattr(reader, name)), reference[f'{case}/{name}']), name
    assert np.allclose(reader.sample.orientationMatrix, reference[f'{case}/UB'],
                       atol=1e-6 if case == 'marmot' else 1e-12) # Marmot: as above
    assert list(reader.scan_parameters) == list(reference[f'{case}/scan_parameters'])
    expected = reference[f'{case}/scan_values']
    if case == 'marmot': # The old code started the scan half a step before the measured A3
        expected = expected + 0.5*np.diff(expected, axis=-1).mean()
    assert np.allclose(reader.scan_values, expected)

# Cuts

@R.needs(R.CAMEA_2024)
@pytest.mark.parametrize('case, options', [('default', {}), ('constant_bins', dict(constant_bins=True)),
                                           ('extend', dict(extend=True))])
def test_cut_1d(camea_2024, case, options):
    reference = R.load('cuts')
    result = camea_2024.cut_1d(HKL1, HKL2, 0.1, 0.02, -1.0, 0.8, **options)
    # Int_err: the old cut1D truncated sqrt(counts) to an integer ("Cut bugs fixed in the port")
    R.assert_frame(reference, f'cut1D/{case}', result.data, skip=['Int_err'])
    positions = result.bins['positions']
    hkl = camea_2024.convert_to_hkl(positions[:, :2])
    assert np.allclose(hkl, reference[f'cut1D/{case}/positions'][:, :3])

@R.needs(R.CAMEA_2024)
@pytest.mark.parametrize('constant_bins', [False, True])
def test_cut_constant_q(constant_bins):
    """The old cut1DE with width w (a radius) is cut_constant_q with width 2w"""
    reference = R.load('cuts')
    energy_min, energy_max = reference['cut1DE/energy']
    result = dataset(R.CAMEA_2024[:1]).cut_constant_q(
        reference['cut1DE/hkl'], width=0.1, min_pixel=0.05, energy_min=energy_min,
        energy_max=energy_max, constant_bins=constant_bins)
    R.assert_frame(reference, f'cut1DE/{constant_bins}', result.data,
                   skip=['BinDistance']) # cut_constant_q has none (as cut_1d)
    assert np.array_equal(result.bins['energy'], reference[f'cut1DE/{constant_bins}/energy_bins'])

@R.needs(R.CAMEA_2024)
def test_cut_qe_line(camea_2024):
    reference = R.load('cuts')
    result = camea_2024.cut_qe_line(QE_LINE, QE_LINE_ENERGY, width=0.1, min_pixel=0.02)
    segments = [segment for _, segment in result.data.groupby('qCut', sort=True)]
    assert len(segments) == reference['cutQELine/segments']
    for index, segment in enumerate(segments):
        R.assert_frame(reference, f'cutQELine/{index}', segment, rtol=1e-12,
                       skip=[c for c in reference[f'cutQELine/{index}/columns']
                             if c not in ['Qx', 'Qy', 'H', 'K', 'L', 'Energy', 'Int', 'intensity',
                                          'BinCount', 'BinDistance']])

@R.needs(R.CAMEA_2024)
def test_cut_powder(camea_2024):
    R.assert_frame(R.load('cuts'), 'cutPowder', camea_2024.cut_powder(POWDER_Q, POWDER_E).data)

@R.needs(R.CAMEA_2024)
@pytest.mark.parametrize('detector, analyser', RAW_SELECTIONS)
def test_cut_raw_1d(detector, analyser):
    reference = R.load('cuts')
    case = f'cutRaw1D/{detector}_{analyser}'
    results = dataset(R.CAMEA_2024).cut_raw_1d(detector, analyser)
    assert list(results[0].parameters['scan_parameters']) == list(reference[f'{case}/parameters'])
    assert list(results[0].parameters['scan_units']) == list(reference[f'{case}/units'])
    for index, result in enumerate(results):
        parameter = result.parameters['scan_parameters'][0]
        assert np.array_equal(result.data[parameter].to_numpy(dtype=float),
                              reference[f'{case}/{index}/x'])
        # A single detector or analyser number reached only the first file in the old code
        if index < reference[f'{case}/files']:
            assert np.array_equal(result.data['intensity'].to_numpy(dtype=float),
                                  reference[f'{case}/{index}/counts'])

@pytest.fixture(scope='module')
def camea_2023_difference(camea_2023):
    return camea_2023 - dataset(R.CAMEA_2023)

@R.needs(R.CAMEA_2023)
@pytest.mark.parametrize('case', ['plain', 'difference'])
def test_cut_qe_and_q_plane(request, camea_2023, case):
    """The milestone script's Q-E cut and Q plane, of the data and of a difference"""
    reference = R.load('cuts')
    data = camea_2023 if case == 'plain' else request.getfixturevalue('camea_2023_difference')
    result = data.cut_qe(QE_Q1, QE_Q2, 0.08, 0.03, energy_min=-0.2,
                         energy_max=float(reference['cutQE/energy_max']), energy_step=0.08)
    # Int_err of a difference: the old cutQE took sqrt of the subtracted counts (NaN where
    # negative); the new cuts use the difference's errors ("Cut bugs fixed in the port")
    R.assert_frame(reference, f'cutQE/{case}', result.data,
                   skip=['Int_err'] if case == 'difference' else [])
    assert np.allclose(result.bins['distance'], reference[f'cutQE/{case}/distance'])
    assert np.allclose(result.bins['energy'], reference[f'cutQE/{case}/energy'])
    assert np.allclose(data.convert_to_hkl(result.bins['orthogonal_extent']),
                       reference[f'cutQE/{case}/orthogonal'])

    result = data.cut_q_plane(energy_min=4.5, energy_max=4.8, x_bin_tolerance=0.03,
                              y_bin_tolerance=0.03)
    # The old plotQPlane gave empty bins as -1 in its DataFrame; Int_err of a difference as above
    R.assert_frame(reference, f'plotQPlane/{case}', result.data.fillna(-1),
                   skip=['Int_err'] if case == 'difference' else [])
    mesh = Plotting.plot_q_plane(result, sample=data.sample).pmeshs[0]
    assert np.array_equal(np.ma.filled(np.asarray(mesh.get_array(), dtype=float), np.nan),
                          reference[f'plotQPlane/{case}/mesh'], equal_nan=True)
    assert np.array_equal(mesh.get_coordinates(), reference[f'plotQPlane/{case}/coordinates'])

@R.needs(R.CAMEA_2023)
@pytest.mark.parametrize('rlu', [True, False])
@pytest.mark.parametrize('steps', BIN3D_STEPS)
def test_bin_3d(camea_2023, rlu, steps):
    reference = R.load('bin3d')
    case = f'{rlu}/{"_".join(map(str, steps))}'
    result = camea_2023.bin_3d(*steps, rlu=rlu)
    for axis in ['qx', 'qy', 'energy']:
        assert np.array_equal(result.bins[axis], reference[f'{case}/{axis}']), axis
    for name in ['intensity', 'Monitor', 'Normalization', 'BinCount']:
        R.assert_summary(reference, f'{case}/{name}', result.grid(name), rtol=0)

# Backgrounds

@R.needs(R.CAMEA_2023)
@pytest.mark.parametrize('combine', [False, True])
def test_powder_background(combine):
    reference = R.load('background')
    data = dataset(R.CAMEA_2023)
    signal = Mask.circleMask(center=[0.0, 0.0], radius=0.6, coordinates=['qx', 'qy'])(data)
    model = data.generate_background_model(background_mask=signal, dQ=0.025, dE=0.05,
                                           combine_foreground_mask=combine)
    for name, values in [('q_bins', model.q_bins), ('energy_bins', model.energy_bins),
                         ('intensity', model.intensity), ('counts', model.counts)]:
        assert np.array_equal(np.asarray(values, dtype=float), reference[f'{combine}/{name}'],
                              equal_nan=True), name
    for index, reader in enumerate(data):
        R.assert_summary(reference, f'{combine}/points/{index}', reader.background)

@R.needs(R.CAMEA_2023)
def test_amber_background():
    """The voxels given to AMBER, and AMBER run with the old model's conventions (lower voxel
    edges, its radial bins and parameters): the new model's own fixes are in test_BackgroundModel"""
    pytest.importorskip('torch')
    background = pytest.importorskip('AMBER.background')
    reference = R.load('amber')
    binned = dataset(R.CAMEA_2023[:int(reference['files'])]).bin_3d(0.03, 0.03, 0.05)
    with np.errstate(all='ignore'):
        intensity = np.divide(binned.grid('intensity')*binned.grid('BinCount'),
                              binned.grid('Monitor')*binned.grid('Normalization'))
    R.assert_summary(reference, 'I', intensity)
    amber = background.background(dtype=intensity.dtype)
    amber.set_gridcell_size(dqx=0.03, dqy=0.03, dE=0.05)
    amber.set_binned_data(Qx=binned.bins['qx'][:-1], Qy=binned.bins['qy'][:-1],
                          E=binned.bins['energy'][:-1], Int=intensity)
    amber.set_radial_bins(max_radius=float(reference['max_radius']), n_bins=int(reference['n_bins']))
    amber.set_variables()
    amber.denoising(amber.Ygrid, lambda_=float(reference['l']), beta_=float(reference['beta']),
                    mu_=float(reference['mu']), n_epochs=int(reference['epochs']), verbose=False)
    R.assert_summary(reference, 'background', amber.b, rtol=1e-9)

# Sample and axes

@R.needs(R.CAMEA_2024)
def test_update_sample_parameters():
    reference = R.load('sample')
    data = dataset(R.CAMEA_2024[:2])
    data.update_sample_parameters(reference['cell'])
    for index, reader in enumerate(data):
        assert np.allclose(reader.sample.UB, reference[f'{index}/UB'], rtol=0, atol=1e-14)
        for name in ['h', 'k', 'l', 'qx', 'qy']:
            R.assert_summary(reference, f'{index}/{name}', getattr(reader, name))

@R.needs(R.CAMEA_2024)
def test_axes(camea_2024):
    reference = R.load('axes')
    for rlu in (True, False):
        ax = camea_2024.create_q_axis(rlu=rlu)
        assert [ax.get_xlabel(), ax.get_ylabel(), ax.format_coord(0.3, 0.2)] == \
            list(reference[f'q/{rlu}'])
        for axis in (0, 1):
            ax = camea_2024.create_qe_axes(axis=axis, rlu=rlu)
            assert [ax.get_xlabel(), ax.format_coord(0.3, 1.0)] == list(reference[f'qe/{rlu}/{axis}'])

# Instrument physics

@R.needs(R.CAMEA_2023)
def test_find_ei_and_second_order_elastic(camea_2023):
    """Within single precision: the old code kept Ei as float32"""
    reference = R.load('physics')
    for index, energy in enumerate(reference['find_ei/energy_transfers']):
        expected = reference[f'find_ei/{index}']
        found = camea_2023.find_ei(energy)
        assert found.shape == expected.shape and np.allclose(found, expected, rtol=1e-6), energy
    for index, atol in enumerate(reference['second_order/atol']):
        assert np.allclose(camea_2023.second_order_elastic(reference['second_order/peaks'], atol=atol),
                           reference[f'second_order/{index}'], rtol=1e-6, equal_nan=True), atol

@R.needs(R.CAMEA_2024)
def test_resolution(camea_2024):
    reference = R.load('physics')
    Ei = float(reference['Ei'])
    projections = {'qx_qy': ([1.0, 0, 0, 0], [0, 1.0, 0, 0]), 'qx_e': ([1.0, 0, 0, 0], [0, 0, 0, 1.0])}
    for i, position in enumerate(reference['resolution/positions']):
        for j, Ef in enumerate(reference['resolution/Ef']):
            expected = reference[f'resolution/{i}/{j}']
            M = camea_2024.resolution_matrix(position, Ei, Ef)
            assert np.allclose(M, expected, rtol=1e-12, atol=1e-12*np.abs(expected).max())
            for name, (first, second) in projections.items():
                for lattice in (False, True):
                    _, axes, sigma = camea_2024.resolution_ellipse(position, first, second, Ei, Ef,
                                                                   lattice_axes=lattice)
                    case = f'ellipse/{i}/{j}/{name}/{lattice}'
                    assert np.allclose(axes, reference[f'{case}/axes'], rtol=1e-9, atol=1e-12)
                    assert np.allclose(sigma, reference[f'{case}/sigma'], rtol=1e-9)
    # Qx, Qy input (the old function needed a sample for it, and failed without one)
    expected = reference['resolution/qxqy/matrix']
    assert np.allclose(camea_2024.resolution_matrix(reference['resolution/qxqy'], Ei, 4.0, rlu=False),
                       expected, rtol=1e-12, atol=1e-12*np.abs(expected).max())
