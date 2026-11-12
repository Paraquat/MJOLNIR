"""Save the old code's output (DataFile.py, DataSet_old.py, BackgroundModel_old.py) as reference
data for test_Reference.py, so the comparisons keep running after the old code is deleted.

Run from the repository root, with the old code still present, and the input files of
test/reference_data.py available:

    python test/reference/generate.py [name ...]

name is one or more of conversion, cuts, bin3d, background, amber, sample, axes (default:
all). amber needs the AMBER-ds4ms package. Each writes test/reference/<name>.npz. The
settings are those of the comparisons made while porting; the values were identical to the
new code's then, apart from the differences test_Reference.py lists.
"""
import sys
import tempfile
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import reference_data as R

from MJOLNIR.Data import DataFile as DataFileModule
from MJOLNIR.Data.DataFile import DataFile
from MJOLNIR.Data.DataSet_old import DataSet as OldDataSet
from MJOLNIR.Data.DataSet import DataSet # Only for inputs the old code takes from the new
from MJOLNIR.Data.ArrayType import ArrayType
from MJOLNIR.Data.InstrumentName import InstrumentName
from MJOLNIR.Data import Mask

warnings.simplefilter('ignore')
quiet = lambda *args, **kwargs: None

def old_dataset(files, binning=8):
    old = OldDataSet(dataFiles=[str(f) for f in files])
    old.convertDataFile(binning=binning, printFunction=quiet)
    return old

def save(name: str, entries: dict):
    path = R.REFERENCE / f'{name}.npz'
    np.savez_compressed(path, **entries)
    print(f'{path.name}: {len(entries)} entries, {path.stat().st_size/1e6:.2f} MB')

# Shared cut settings
HKL1, HKL2 = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
QE_Q1, QE_Q2 = np.array([-0.5, 1.5, 0.0]), np.array([1.5, -0.5, 0.0]) # The milestone script
QE_LINE = np.array([[-1, 0, 0], [0, 1, 0], [0.5, 0.5, 0], [0, 0, 0]], dtype=float)
QE_LINE_ENERGY = np.arange(-1.0, 0.81, 0.1)
POWDER_Q, POWDER_E = np.arange(0.4, 2.0, 0.05), np.arange(-1, 0.81, 0.1)
RAW_SELECTIONS = [(None, None), (0, 0), (50, 3), (103, 7)]
BIN3D_STEPS = [(0.05, 0.05, 0.1), (0.08, 0.03, 0.25)]

def conversion():
    """Converted arrays per instrument, in the readers' shape (steps, detectors, bins)"""
    entries = {}

    def store(case, old, converted, shape):
        for name in ['I', 'Norm', 'Monitor', 'qx', 'qy', 'energy', 'h', 'k', 'l']:
            entries.update(R.summary(f'{case}/{name}',
                                     np.asarray(getattr(converted, name), dtype=float).reshape(shape)))
        entries[f'{case}/A3'] = np.asarray(old.A3, dtype=float).ravel()
        entries[f'{case}/A4'] = np.asarray(old.A4, dtype=float).ravel()
        entries[f'{case}/Ei'] = np.asarray(old.Ei, dtype=float).ravel()
        entries[f'{case}/UB'] = np.asarray(old.sample.orientationMatrix, dtype=float)
        entries[f'{case}/scan_parameters'] = np.array(list(old.scanParameters), dtype=str)
        entries[f'{case}/scan_values'] = np.asarray(old.scanValues, dtype=float)

    for case, path, binning, offset in [('camea2024_b8', R.CAMEA_2024[0], 8, None),
                                        ('camea2024_b3_a3offset', R.CAMEA_2024[0], 3, 1.5),
                                        ('camea2023_b8', R.CAMEA_2023[0], 8, None)]:
        old = DataFile(str(path))
        if offset is not None:
            old.A3Off = np.array([offset])
        converted = old.convert(binning, printFunction=quiet)
        store(case, old, converted, converted.I.shape)
    for case, path, shape in [('bambus', R.BAMBUS, (-1, 20, 5)),
                              ('flatcone', R.FLATCONE, (-1, 31, 1))]:
        old = DataFile(str(path))
        store(case, old, old.convert(1, printFunction=quiet), shape)
    with tempfile.TemporaryDirectory() as folder:
        old = DataFile(str(R.multiflexx_file(Path(folder) / 'multiflexx_synthetic')))
        store('multiflexx', old, old.convert(1, printFunction=quiet), (-1, 31, 5))
        # The old Marmot loader is only reachable by hand (as test_MarmotDataReader's table)
        old = DataFile.__new__(DataFile)
        old.fileLocation, old.name, old.type = str(R.MARMOT), R.MARMOT.name, 'Marmot'
        old._binning, old._mask = 1, False
        DataFileModule.loadMarmot(old, str(R.marmot_table(Path(folder) / 'marmot.table')))
        store('marmot', old, old.convert(1, printFunction=quiet), (-1, 30, 256))
    save('conversion', entries)

def cuts():
    entries = {}
    old = old_dataset(R.CAMEA_2024[:2])
    for case, options in [('default', {}), ('constant_bins', dict(constantBins=True)),
                          ('extend', dict(extend=True))]:
        data, bins = old.cut1D(HKL1, HKL2, width=0.1, minPixel=0.02, EMin=-1.0, EMax=0.8, **options)
        entries.update(R.frame(f'cut1D/{case}', data))
        entries[f'cut1D/{case}/positions'] = np.asarray(bins[0], dtype=float)

    data_list, _ = old.cutQELine(QE_LINE, QE_LINE_ENERGY, width=0.1, minPixel=0.02, rlu=True)
    entries['cutQELine/segments'] = np.array(len(data_list))
    for index, data in enumerate(data_list):
        entries.update(R.frame(f'cutQELine/{index}', data))
    entries.update(R.frame('cutPowder', old.cutPowder(POWDER_E, POWDER_Q)))

    # cut1DE on one file, at a measured point (as test_CutConstantQ chose it)
    old = old_dataset(R.CAMEA_2024[:1])
    new = DataSet(InstrumentName.CAMEA, files=R.CAMEA_2024[:1])
    qx, qy = np.asarray(new.gather(ArrayType.QX)), np.asarray(new.gather(ArrayType.QY))
    energy = np.asarray(new.gather(ArrayType.ENERGY))
    i = len(qx)//3
    hkl = new.convert_to_hkl(np.array([qx[i], qy[i]]))
    energy_min, energy_max = float(np.percentile(energy, 10)), float(np.percentile(energy, 90))
    entries['cut1DE/hkl'] = hkl
    entries['cut1DE/energy'] = np.array([energy_min, energy_max])
    for constant in (False, True):
        data, bins = old.cut1DE(E1=energy_min, E2=energy_max, q=hkl, width=0.05, minPixel=0.05,
                                constantBins=constant)
        entries.update(R.frame(f'cut1DE/{constant}', data))
        entries[f'cut1DE/{constant}/energy_bins'] = np.asarray(bins[0][0], dtype=float)

    raw = OldDataSet(dataFiles=[str(f) for f in R.CAMEA_2024]) # Raw counts: no conversion
    for detector, analyser in RAW_SELECTIONS:
        X, I, parameters, units = raw.cutRaw1D(detectorSelection=detector, analyzerSelection=analyser)
        case = f'cutRaw1D/{detector}_{analyser}'
        entries[f'{case}/parameters'] = np.array(list(parameters), dtype=str)
        entries[f'{case}/units'] = np.array(list(units), dtype=str)
        # A single detector or analyser number only reached the first file (the old code
        # zipped the files with a one-element list): its counts are saved for that file only
        entries[f'{case}/files'] = np.array(len(I))
        for index, x in enumerate(X):
            entries[f'{case}/{index}/x'] = np.asarray(x[0], dtype=float)
        for index, counts in enumerate(I):
            entries[f'{case}/{index}/counts'] = np.asarray(counts, dtype=float)

    # The milestone script's Q-E cut and Q plane (camea_test), also of a difference
    old, other = old_dataset(R.CAMEA_2023), old_dataset(R.CAMEA_2023)
    energy_max = float(old.energy.max())
    entries['cutQE/energy_max'] = np.array(energy_max)
    for case, dataset in [('plain', old), ('difference', old - other)]:
        data, bins, ortho = dataset.cutQE(QE_Q1, QE_Q2, width=0.08, minPixel=0.03, EMin=-0.2,
                                          EMax=energy_max, dE=0.08)
        entries.update(R.frame(f'cutQE/{case}', data))
        entries[f'cutQE/{case}/distance'] = np.asarray(bins[0][:, 0], dtype=float)
        entries[f'cutQE/{case}/energy'] = np.asarray(bins[1][0, :], dtype=float)
        entries[f'cutQE/{case}/orthogonal'] = np.asarray(ortho, dtype=float)
        data, ax = dataset.plotQPlane(EMin=4.5, EMax=4.8, xBinTolerance=0.03, yBinTolerance=0.03)
        entries.update(R.frame(f'plotQPlane/{case}', data))
        mesh = ax.pmeshs[0]
        entries[f'plotQPlane/{case}/mesh'] = np.ma.filled(np.asarray(mesh.get_array(), dtype=float), np.nan)
        entries[f'plotQPlane/{case}/coordinates'] = np.asarray(mesh.get_coordinates(), dtype=float)
        plt.close('all')
    save('cuts', entries)

def bin3d():
    entries = {}
    old = old_dataset(R.CAMEA_2023)
    for rlu in (True, False):
        for steps in BIN3D_STEPS:
            (I, monitor, normalization, count), bins = old.binData3D(*steps, rlu=rlu)
            case = f'{rlu}/{"_".join(map(str, steps))}'
            for axis, edges in zip(['qx', 'qy', 'energy'],
                                   [bins[0][:, 0, 0], bins[1][0, :, 0], bins[2][0, 0, :]]):
                entries[f'{case}/{axis}'] = np.asarray(edges, dtype=float)
            for name, array in [('intensity', I), ('Monitor', monitor),
                                ('Normalization', normalization), ('BinCount', count)]:
                entries.update(R.summary(f'{case}/{name}', array))
    save('bin3d', entries)

def background():
    """The powder background with a signal mask (a circle around Q = 0 in Qx, Qy, evaluated by
    the new Mask.py, which the old code takes as arrays)"""
    entries = {}
    new = DataSet(InstrumentName.CAMEA, files=R.CAMEA_2023)
    signal = Mask.circleMask(center=[0.0, 0.0], radius=0.6, coordinates=['qx', 'qy'])(new)
    for combine in (False, True):
        old = old_dataset(R.CAMEA_2023)
        old.generateBackgroundModel(backgroundMask=[m.copy() for m in signal], dQ=0.025, dE=0.05,
                                    combineForegroundMask=combine)
        model = old.backgroundModel
        for name, array in [('q_bins', model.QBins), ('energy_bins', model.EnergyBins),
                            ('intensity', model.int), ('counts', model.intensity)]:
            entries[f'{combine}/{name}'] = np.asarray(array, dtype=float)
        _ = old.backgroundIntensities # Computes the per-point background
        for index, data_file in enumerate(old):
            entries.update(R.summary(f'{combine}/points/{index}', data_file.backgroundIntensities))
    save('background', entries)

def amber():
    """AMBER on the first two camea_test files, 5 iterations: the voxel intensities the old model
    gave AMBER and its background, with the old conventions"""
    from MJOLNIR.Data import BackgroundModel_old
    files = R.CAMEA_2023[:2]
    old = old_dataset(files)
    model = BackgroundModel_old.AMBERBackground(old, dQx=0.03, dQy=0.03, dE=0.05)
    model.set_radial_bins()
    model.generateAMBER(n_epochs=5, verbose=False)
    entries = R.summary('I', model.I)
    entries.update(R.summary('background', model.AMBER.b))
    for name in ['max_radius', 'n_bins', 'l', 'mu', 'beta']:
        entries[name] = np.array(getattr(model, name), dtype=float)
    entries['files'] = np.array(len(files))
    entries['epochs'] = np.array(5)
    save('amber', entries)

def sample():
    """updateSampleParameters: a and b 1% longer, gamma 91 degrees, then converted"""
    entries = {}
    old = OldDataSet(dataFiles=[str(f) for f in R.CAMEA_2024[:2]])
    cell = np.asarray(old[0].sample.unitCell, dtype=float).copy()
    cell[:2] *= 1.01
    cell[5] = 91.0
    entries['cell'] = cell
    old.updateSampleParameters(cell)
    old.convertDataFile(binning=8, printFunction=quiet)
    for index, converted in enumerate(old.convertedFiles):
        entries[f'{index}/UB'] = np.asarray(converted.sample.UB, dtype=float)
        for name in ['h', 'k', 'l', 'qx', 'qy']:
            entries.update(R.summary(f'{index}/{name}', getattr(converted, name)))
    save('sample', entries)

def axes():
    """Labels and read-out of the old createQAxis and createQEAxes"""
    entries = {}
    old = old_dataset(R.CAMEA_2024[:2])
    for rlu in (True, False):
        ax = old.createQAxis(rlu=rlu, withoutOnClick=True)
        entries[f'q/{rlu}'] = np.array([ax.get_xlabel(), ax.get_ylabel(), ax.format_coord(0.3, 0.2)])
        for axis in (0, 1):
            ax = old.createQEAxes(axis=axis, rlu=rlu)
            entries[f'qe/{rlu}/{axis}'] = np.array([ax.get_xlabel(), ax.format_coord(0.3, 1.0)])
        plt.close('all')
    save('axes', entries)

GENERATORS = dict(conversion=conversion, cuts=cuts, bin3d=bin3d, background=background,
                  amber=amber, sample=sample, axes=axes)

if __name__ == '__main__':
    for name in sys.argv[1:] or GENERATORS:
        GENERATORS[name]()
