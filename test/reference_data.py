"""Reference outputs of the old code (DataFile.py, DataSet_old.py, BackgroundModel_old.py),
saved by test/reference/generate.py before the old code was deleted (commit 15909fd is the
last with it), and helpers to compare
the new code with them (test_Reference.py).

Arrays too large to store (converted data, 3D bins, per-point backgrounds) are kept as a
summary: shape, number of NaN, sum, sum of squares, minimum, maximum and the values at up to
SAMPLE_SIZE fixed positions. Cut results are stored whole, one array per column.
"""
from pathlib import Path
import numpy as np
import pytest

REFERENCE = Path(__file__).parent / 'reference'
SAMPLE_SIZE = 5000

# Input data (not in the repository yet; question 36 of the refactoring log)
SAMPLE_DATA = Path(__file__).parent.parent / 'sampledata'
PROJECTS = Path.home() / 'projects'
CAMEA_2024 = sorted(SAMPLE_DATA.glob('camea2024*.hdf'))
CAMEA_2023 = sorted((PROJECTS / 'camea_test').glob('camea2023*.hdf'))
BAMBUS = PROJECTS / 'CoRhO4' / 'corh2o4_test.dat'
FLATCONE = PROJECTS / 'marmot+flatcone' / '043555_pastis.nxs'
MARMOT = PROJECTS / 'marmot+flatcone' / '054724.nxs'

def needs(*paths):
    """Skip unless every input file exists (a list counts as missing when empty)"""
    missing = [p for p in paths if (isinstance(p, list) and not p)
               or (not isinstance(p, list) and not Path(p).exists())]
    return pytest.mark.skipif(bool(missing), reason=f'input data not available: {missing}')

def load(name: str) -> dict:
    """The arrays of reference file name (without extension)"""
    with np.load(REFERENCE / f'{name}.npz', allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def sample_positions(size: int) -> np.ndarray:
    """The fixed positions sampled from an array of size elements"""
    if size <= SAMPLE_SIZE:
        return np.arange(size)
    return np.sort(np.random.default_rng(0).choice(size, SAMPLE_SIZE, replace=False))

def summary(prefix: str, array) -> dict:
    """The summary of array, as entries prefix/... of a reference file"""
    array = np.asarray(array, dtype=float)
    finite = array[np.isfinite(array)]
    return {f'{prefix}/shape': np.array(array.shape),
            f'{prefix}/nan': np.array(np.isnan(array).sum()),
            f'{prefix}/sum': np.array(finite.sum()),
            f'{prefix}/sum2': np.array(np.square(finite).sum()),
            f'{prefix}/min': np.array(finite.min() if finite.size else np.nan),
            f'{prefix}/max': np.array(finite.max() if finite.size else np.nan),
            f'{prefix}/sample': array.ravel()[sample_positions(array.size)]}

def assert_summary(reference: dict, prefix: str, array, rtol: float = 1e-12, atol: float = 0.0):
    """array matches the summary saved under prefix: the sampled values within rtol and atol,
    the sums within rtol or 1e-10, whichever is larger"""
    array = np.asarray(array, dtype=float)
    assert tuple(array.shape) == tuple(reference[f'{prefix}/shape']), prefix
    assert np.isnan(array).sum() == reference[f'{prefix}/nan'], prefix
    finite = array[np.isfinite(array)]
    for key, value in [('sum', finite.sum()), ('sum2', np.square(finite).sum()),
                       ('min', finite.min() if finite.size else np.nan),
                       ('max', finite.max() if finite.size else np.nan)]:
        assert np.allclose(value, reference[f'{prefix}/{key}'], rtol=max(rtol, 1e-10),
                           equal_nan=True), \
            f'{prefix}: {key}'
    assert np.allclose(array.ravel()[sample_positions(array.size)],
                       reference[f'{prefix}/sample'], rtol=rtol, atol=atol, equal_nan=True), prefix

def frame(prefix: str, data) -> dict:
    """The columns of a DataFrame, as entries prefix/<column> (and prefix/columns)"""
    entries = {f'{prefix}/columns': np.array(list(data.columns), dtype=str)}
    for column in data.columns:
        entries[f'{prefix}/{column}'] = np.asarray(data[column], dtype=float)
    return entries

def assert_frame(reference: dict, prefix: str, data, skip=(), rtol: float = 0.0):
    """Every saved column (but skip) of a DataFrame matches; exactly with rtol 0"""
    columns = [c for c in reference[f'{prefix}/columns'] if c not in skip]
    assert len(data) == len(reference[f'{prefix}/{columns[0]}']), f'{prefix}: rows'
    for column in columns:
        values = np.asarray(data[column], dtype=float)
        expected = reference[f'{prefix}/{column}']
        if rtol == 0:
            assert np.array_equal(values, expected, equal_nan=True), f'{prefix}/{column}'
        else:
            assert np.allclose(values, expected, rtol=rtol, atol=0, equal_nan=True), \
                f'{prefix}/{column}'

def marmot_table(path: Path) -> Path:
    """The placeholder Marmot calibration of test_MarmotDataReader (none is shipped): one
    energy bin per pixel (Ef 1 to 5 meV), detectors 2 degrees apart"""
    rows = [[det, pix, 0, 1.0, 1.0 + 4.0*pix/255, 0.1, 0.0, pix, pix + 1, 2.0*det]
            for det in range(30) for pix in range(256)]
    np.savetxt(path, rows, delimiter=',',
               header='Detector,Energy,Pixel,Amplitude,Center,Width,Background,lowerBin,upperBin,A4Offset')
    return path

def multiflexx_file(path: Path) -> Path:
    """A synthetic MultiFLEXX file (none is available): the Bambus file's header with a 'mira'
    instrument line, 21 steps of sth and 155 channels of random counts (seeded)"""
    source = BAMBUS.read_text().splitlines()
    end = next(i for i, line in enumerate(source) if '### Scan data' in line)
    header = source[:end + 1]
    header.insert(header.index('### Instrument setup') + 1, '#                 mira_alias : mira')
    rng = np.random.default_rng(1)
    columns = ['sth', 'stt', 'chsum'] + [f'ch{i}' for i in range(155)] + ['events']
    lines = header + ['#  ' + ' '.join(columns), '#  ' + ' '.join(['deg', 'deg'] + ['cts']*157)]
    for sth in np.arange(-20, 21, 2.0):
        lines.append(' '.join([f'{sth:.3f}', '60.500', '0']
                              + [str(x) for x in rng.integers(0, 50, 155)] + ['0']))
    lines.append('### End of NICOS data file')
    path.write_text('\n'.join(lines) + '\n')
    return path
