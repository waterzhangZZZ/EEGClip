"""
Basic Brain Decoding on EEG Data
========================================

This tutorial shows you how to train and test deep learning models with
Braindecode in a classical EEG setting: you have trials of data with
labels (e.g., Right Hand, Left Hand, etc.).

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Loading and preparing the data
# -------------------------------------
#


######################################################################
# Loading the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# First, we load the data. In this tutorial, we load the BCI Competition
# IV 2a data [1]_ using braindecode's wrapper to load via
# `MOABB library <https://github.com/NeuroTechX/moabb>`__ [2]_.
#
# .. note::
#    To load your own datasets either via mne or from
#    preprocessed X/y numpy arrays, see `MNE Dataset
#    Tutorial <./plot_mne_dataset_example.html>`__ and `Numpy Dataset
#    Tutorial <./plot_custom_dataset_example.html>`__.
#

from braindecode.datasets import MOABBDataset

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])


######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#


######################################################################
# Now we apply preprocessing like bandpass filtering to our dataset. You
# can either apply functions provided by
# `mne.Raw <https://mne.tools/stable/generated/mne.io.Raw.html>`__ or
# `mne.Epochs <https://mne.tools/0.11/generated/mne.Epochs.html#mne.Epochs>`__
# or apply your own functions, either to the MNE object or the underlying
# numpy array.
#
# .. note::
#    Generally, braindecode prepocessing is directly applied to the loaded
#    data, and not applied on-the-fly as transformations, such as in
#    PyTorch-libraries like
#    `torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`__.
#

from numpy import multiply

from braindecode.preprocessing import (Preprocessor,
                                       exponential_moving_standardize,
                                       preprocess)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors, n_jobs=-1)


######################################################################
# Extracting Compute Windows
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Now we extract compute windows from the signals, these will be the inputs
# to the deep networks during training. In the case of trialwise
# decoding, we just have to decide if we want to include some part
# before and/or after the trial. For our work with this dataset,
# it was often beneficial to also include the 500 ms before the trial.
#

from braindecode.preprocessing import create_windows_from_events

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)


######################################################################
# Splitting the dataset into training and validation sets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# We can easily split the dataset using additional info stored in the
# description attribute, in this case ``session`` column. We select
# ``T`` for training and ``test`` for validation.
#

splitted = windows_dataset.split('session')
train_set = splitted['0train']  # Session train
valid_set = splitted['1test']  # Session evaluation


######################################################################
# Creating a model
# ------------
#


######################################################################
# Now we create the deep learning model! Braindecode comes with some
# predefined convolutional neural network architectures for raw
# time-domain EEG. Here, we use the shallow ConvNet model from [3]_. These models are
# pure `PyTorch <https://pytorch.org>`__ deep learning models, therefore
# to use your own model, it just has to be a normal PyTorch
# `nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__.
#

import torch

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True
# Set random seed to be able to roughly reproduce results
# Note that with cudnn benchmark set to True, GPU indeterminism
# may still make results substantially different between runs.
# To obtain more consistent results at the cost of increased computation time,
# you can set `cudnn_benchmark=False` in `set_random_seeds`
# or remove `torch.backends.cudnn.benchmark = True`
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = train_set[0][0].shape[1]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    model = model.cuda()


######################################################################
# Model Training
# --------------
#


######################################################################
# Now we will train the network! ``EEGClassifier`` is a Braindecode object
# responsible for managing the training of neural networks. It inherits
# from skorch `NeuralNetClassifier <https://skorch.readthedocs.io/en/stable/classifier.html#>`__,
# so the training logic is the same as in `Skorch <https://skorch.readthedocs.io/en/stable/>`__.
#


######################################################################
# .. note::
#    In this tutorial, we use some default parameters that we
#    have found to work well for motor decoding, however we strongly
#    encourage you to perform your own hyperparameter optimization using
#    cross validation on your training data.
#

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier

# We found these values to be good for the shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 4

clf = EEGClassifier(
    model,
    criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
    classes=classes,
)
# Model training for the specified number of epochs. `y` is None as it is
# already supplied in the dataset.
_ = clf.fit(train_set, y=None, epochs=n_epochs)


from braindecode.visualization.gradients import compute_amplitude_gradients

amp_gradients = compute_amplitude_gradients(model, windows_dataset, batch_size=16)

print(f"Amplitude gradients shape: {amp_gradients.shape}")

import numpy as np
np.save("amplitude_gradients.npy", amp_gradients)