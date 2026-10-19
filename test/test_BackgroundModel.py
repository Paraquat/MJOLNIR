import pytest

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MJOLNIR.Data import Mask, Plotting
from MJOLNIR.Data.BackgroundModel import PowderBackground, AMBERBackground
from MJOLNIR.Data.DataSet import DataSet, ArrayType
from MJOLNIR.Data.InstrumentName import InstrumentName

files = sorted((Path(__file__).parent.parent / 'sampledata').glob('camea2024*.hdf'))

# The model on its own, from arrays

def test_bin_sizes():
    model = PowderBackground()
    assert (model.dQ, model.dE) == (0.025, 0.05)
    with pytest.raises(AttributeError):
        PowderBackground(dQ=0.0)
    with pytest.raises(AttributeError):
        model.dE = -0.1
    with pytest.raises(AttributeError): # Not fitted yet
        model.sample([1.0], [1.0])

@pytest.fixture
def fitted():
    # Two points in one bin, one in another; normalization 2 and monitor 10 everywhere
    q_length = np.array([0.31, 0.32, 0.71])
    energy = np.array([1.01, 1.02, 2.01])
    intensity = np.array([4.0, 6.0, 3.0])
    ones = np.ones(3)
    return PowderBackground(dQ=0.1, dE=0.1).fit(q_length, energy, intensity, 2*ones, 10*ones)

def test_fit(fitted):
    # |Q| bins start at 0, energy bins extend 1.1 bins beyond the data
    assert fitted.q_bins[0] == 0.0
    assert fitted.q_bins[-1] > 0.71
    assert fitted.energy_bins[0] == pytest.approx(1.01 - 0.11)
    assert fitted.intensity.shape == (len(fitted.q_bins) - 1, len(fitted.energy_bins) - 1)
    # Normalised like a cut: counts * points / (normalization * monitor)
    assert np.nansum(fitted.bin_count) == 3
    assert np.nanmax(fitted.intensity) == pytest.approx(10 * 2 / (4 * 20))
    assert np.isnan(fitted.intensity).sum() == fitted.intensity.size - 2 # Empty bins

def test_sample(fitted):
    intensity = fitted.sample([0.35, 0.75], [1.05, 2.05])
    assert intensity == pytest.approx([10 * 2 / (4 * 20), 3 / 20])
    # With normalization and monitor: the counts each point would have recorded
    counts = fitted.sample([0.35, 0.75], [1.05, 2.05], normalization=[2, 1], monitor=[10, 10])
    assert counts == pytest.approx([5.0, 1.5])
    # Empty bin inside the range: NaN
    assert np.isnan(fitted.sample([0.55], [1.55]))[0]

def test_sample_outside(fitted):
    # Outside the binned range: 0. Below the first edge used to wrap to the last bin
    below = fitted.sample([0.35, 0.35], [0.5, -5.0])
    above = fitted.sample([5.0, 0.35], [1.05, 50.0])
    assert np.array_equal(below, [0.0, 0.0])
    assert np.array_equal(above, [0.0, 0.0])

# In a DataSet

signal = Mask.circleMask(center=[-1, 0], radius=0.25, coordinates=['h', 'k'])

@pytest.fixture
def dataset():
    return DataSet(InstrumentName.CAMEA, files=files[:2])

def test_generate(dataset):
    model = dataset.generate_background_model(background_mask=signal)
    assert dataset.background_model is model
    for reader in dataset:
        assert reader.background.shape == reader.intensity.shape
    # The fit leaves out the signal mask instead of the DataSet's mask, which is unchanged
    assert np.nansum(model.bin_count) == sum((~m).sum() for m in signal(dataset))
    assert all(m.sum() == 135 * 104 * 2 for m in dataset.mask) # Default mask only
    # The per-point background is the model at each point's |Q| and energy
    reader = dataset[0]
    q_length = np.linalg.norm([reader.qx.flatten(), reader.qy.flatten()], axis=0)
    expected = model.sample(q_length, reader.energy.flatten(), reader.normalization.flatten(),
                            reader.monitor.flatten())
    assert np.array_equal(reader.background.flatten(), expected, equal_nan=True)
    view = dataset.get_view(ArrayType.BACKGROUND)
    assert view.shape == (270, 104, 64)
    assert len(dataset.gather(ArrayType.BACKGROUND)) == len(dataset.gather(ArrayType.INTENSITY))

def test_combine_foreground_mask(dataset):
    alone = dataset.generate_background_model(background_mask=signal)
    combined = dataset.generate_background_model(background_mask=signal,
                                                 combine_foreground_mask=True)
    # The default mask (2 of 64 pixels per detector) is left out as well
    union = sum((s | m).sum() for s, m in zip(signal(dataset), dataset.mask))
    assert np.nansum(combined.bin_count) == sum(m.size for m in dataset.mask) - union
    assert np.nansum(combined.bin_count) < np.nansum(alone.bin_count)
    # Default: the DataSet's own mask
    own = dataset.generate_background_model()
    assert np.nansum(own.bin_count) == sum((~m).sum() for m in dataset.mask)

def test_remove(dataset):
    dataset.generate_background_model(background_mask=signal)
    dataset.remove_background_model()
    assert dataset.background_model is None
    assert all(reader.background is None for reader in dataset)

def test_follows_binning(dataset):
    dataset.generate_background_model(background_mask=signal, dQ=0.05)
    dataset.set_binning(3) # A mask object is evaluated again: the model is refitted
    assert dataset[0].background.shape == (135, 104, 24)
    assert dataset.background_model.dQ == 0.05
    # Arrays do not fit another binning: the model is removed
    dataset.generate_background_model(background_mask=signal(dataset))
    with pytest.warns(UserWarning, match='removed'):
        dataset.set_binning(8)
    assert dataset.background_model is None
    assert all(reader.background is None for reader in dataset)

def test_append_and_slice(dataset):
    model = dataset.generate_background_model(background_mask=signal)
    dataset.append(files[2]) # Added files get their background from the existing model
    assert dataset[2].background.shape == dataset[2].intensity.shape
    assert dataset.background_model is model
    assert dataset[:2].background_model is model

def test_difference_and_frozen(dataset):
    dataset.generate_background_model(background_mask=signal)
    difference = dataset - DataSet(InstrumentName.CAMEA, files=files[:2])
    assert all(reader.background is None for reader in difference)
    assert difference.background_model is None
    with pytest.raises(Exception, match='background'):
        difference.generate_background_model()

def test_plot(dataset):
    model = dataset.generate_background_model(background_mask=signal)
    ax = Plotting.plot_powder_background(model)
    assert ax.get_xlabel() == '|Q| [1/A]'
    assert len(ax.collections) == 1
    plt.close(ax.get_figure())

def test_cut_subtraction(dataset):
    with pytest.raises(AttributeError, match='background model'):
        dataset.cut_1d([-1, 0, 0], [0, 1, 0], 0.1, 0.02, -1.0, 0.8, background_subtraction=True)
    dataset.generate_background_model(background_mask=signal)
    cut = dataset.cut_1d([-1, 0, 0], [0, 1, 0], 0.1, 0.02, -1.0, 0.8,
                         background_subtraction=True).data
    plain = dataset.cut_1d([-1, 0, 0], [0, 1, 0], 0.1, 0.02, -1.0, 0.8).data
    assert np.array_equal(cut['Int_Fg'], plain['Int'], equal_nan=True)
    assert np.allclose(cut['Int'], cut['Int_Fg'] - cut['Int_Bg'], equal_nan=True)
    assert np.nansum(cut['intensity_Bg']) > 0
    plane = dataset.cut_q_plane(energy_min=-0.2, energy_max=0.2, background_subtraction=True)
    assert 'Int_Bg' in plane.data.columns
    ax = dataset.plot_cut_qe([-1, 0, 0], [0, 1, 0], energy_min=-1.0, energy_max=0.8,
                             energy_step=0.1, background_subtraction=True)
    assert 'Int_Bg' in ax.result.data.columns
    plt.close(ax.get_figure())

def test_bin_3d_subtraction(dataset):
    dataset.generate_background_model(background_mask=signal)
    result = dataset.bin_3d(0.1, 0.1, 0.5, background_subtraction=True)
    assert np.nansum(result.data['intensity_Bg']) > 0
    assert np.allclose(result.data['Int'], result.data['Int_Fg'] - result.data['Int_Bg'],
                       equal_nan=True)

# AMBER (the optional AMBER-ds4ms package)

def test_amber_missing(monkeypatch, dataset):
    import sys
    monkeypatch.setitem(sys.modules, 'AMBER', None) # As if not installed
    with pytest.raises(ModuleNotFoundError, match='MJOLNIR\\[AMBER\\]'):
        dataset.generate_amber_background(n_epochs=1, verbose=False)

@pytest.fixture
def amber_dataset():
    pytest.importorskip('AMBER')
    return DataSet(InstrumentName.CAMEA, files=files[:1])

def test_amber(amber_dataset):
    binned = amber_dataset.bin_3d(0.1, 0.1, 0.2)
    model = AMBERBackground().fit(binned, n_epochs=3, verbose=False)
    parameters = model.fitted_parameters
    # Radial bins from 0, about as wide as the voxels, covering every voxel with data
    assert model.q_bins[0] == 0.0
    assert len(model.q_bins) == parameters['n_bins'] + 1
    assert model.q_bins[-1] == pytest.approx(parameters['max_radius'])
    assert np.diff(model.q_bins)[0] == pytest.approx(0.1, rel=0.1)
    q_length = np.linalg.norm([amber_dataset.gather(ArrayType.QX),
                               amber_dataset.gather(ArrayType.QY)], axis=0)
    assert q_length.max() <= model.q_bins[-1] # Aligning to the lattice keeps |Q|
    assert np.array_equal(model.energy_bins, binned.bins['energy'])
    assert model.intensity.shape == (len(model.q_bins) - 1, len(model.energy_bins) - 1)
    assert (model.intensity >= 0).all()
    assert model.amber.X.shape == model.amber.Ygrid.shape # The signal
    # Given parameters are used, and the model keeps them as given
    model = AMBERBackground(l=0.01, n_bins=10).fit(binned, n_epochs=2, verbose=False)
    assert model.fitted_parameters['l'] == 0.01
    assert len(model.q_bins) == 11
    assert model.l == 0.01 and model.mu is None

def test_amber_in_dataset(amber_dataset):
    model = amber_dataset.generate_amber_background(dQx=0.1, dQy=0.1, dE=0.2, n_epochs=2,
                                                    verbose=False)
    assert isinstance(amber_dataset.background_model, AMBERBackground)
    reader = amber_dataset[0]
    assert reader.background.shape == reader.intensity.shape
    assert np.nansum(reader.background) > 0
    cut = amber_dataset.cut_1d([-1, 0, 0], [0, 1, 0], 0.1, 0.02, -1.0, 0.8,
                               background_subtraction=True).data
    assert np.nansum(cut['intensity_Bg']) > 0
    amber_dataset.set_binning(3) # Refitted with the same settings
    assert amber_dataset.background_model is not model
    assert isinstance(amber_dataset.background_model, AMBERBackground)
    assert reader.background.shape == (reader.intensity.shape)

def test_amber_cross_validation(amber_dataset):
    binned = amber_dataset.bin_3d(0.1, 0.1, 0.2)
    model = AMBERBackground()
    rmse = model.cross_validation(binned, beta=[0.5, 2.0], n_epochs=2, verbose=False)
    assert rmse.shape == (2,)
    assert np.all(np.isfinite(rmse))
    assert model.q_bins is None # Not fitted
