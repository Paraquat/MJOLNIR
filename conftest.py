import os
from pathlib import Path

import pytest

collect_ignore_glob = ["MJOLNIR/CommandLineScripts/MJOLNIRHistory.py","MJOLNIR/CommandLineScripts/MJOLNIR3DView.py","MJOLNIR/CommandLineScripts/MJOLNIRCalibrationInspector.py"]

DATA_DIR_ENV = 'MJOLNIR_TEST_DATA'
DEFAULT_DATA_DIR = Path(__file__).parent / 'sampledata'


def pytest_addoption(parser):
    # Registered in the root conftest so the option is known however pytest is invoked
    parser.addoption('--data-dir', default=None, metavar='DIR',
                     help=f'Directory with the test input data (default: ${DATA_DIR_ENV}, '
                          f'else <repository>/sampledata)')


def resolve_data_dir(config: pytest.Config) -> Path:
    """The test data directory: --data-dir, else $MJOLNIR_TEST_DATA, else <repository>/sampledata.
    Usable at collection time (pytest_generate_tests) as well as in fixtures."""
    value = config.getoption('data_dir') or os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR
    return Path(value).expanduser().resolve()


@pytest.fixture(scope='session')
def data_dir(pytestconfig) -> Path:
    """The test data directory (see resolve_data_dir)"""
    return resolve_data_dir(pytestconfig)


def pytest_report_header(config):
    data = resolve_data_dir(config)
    return f'test data: {data}' + ('' if data.is_dir() else ' (missing)')


# Test input files. Each fixture skips the requesting test when its data is not in data_dir.

def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f'test data not available: {path}')
    return path


def _glob(data_dir: Path, pattern: str) -> list:
    files = sorted(data_dir.glob(pattern))
    if not files:
        pytest.skip(f'test data not available: {data_dir / pattern}')
    return files


@pytest.fixture(scope='session')
def camea_2024_files(data_dir) -> list:
    """The CAMEA 2024 example scans (camea2024*.hdf), sorted"""
    return _glob(data_dir, 'camea2024*.hdf')


@pytest.fixture(scope='session')
def camea_2023_files(data_dir) -> list:
    """The CAMEA 2023 test scans (camea2023*.hdf), sorted"""
    return _glob(data_dir, 'camea2023*.hdf')


@pytest.fixture(scope='session')
def bambus_file(data_dir) -> Path:
    return _require(data_dir / 'corh2o4_test.dat')


@pytest.fixture(scope='session')
def flatcone_file(data_dir) -> Path:
    return _require(data_dir / '043555_pastis.nxs')


@pytest.fixture(scope='session')
def marmot_file(data_dir) -> Path:
    return _require(data_dir / '054724.nxs')


def pytest_generate_tests(metafunc):
    # camea_2024_file: one test per CAMEA 2024 scan in the data directory (after option parsing)
    if 'camea_2024_file' in metafunc.fixturenames:
        files = sorted(resolve_data_dir(metafunc.config).glob('camea2024*.hdf'))
        params = [pytest.param(f, id=f.name) for f in files] or [
            pytest.param(None, id='no-data', marks=pytest.mark.skip(reason='test data not available: camea2024*.hdf'))]
        metafunc.parametrize('camea_2024_file', params)
