# %%
# mne imports
import mne
import numpy as np
from keras import backend as K
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from mne import io
from mne.datasets import sample
# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# %%
# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format("channels_last")

# %%
##################### Process, filter and epoch the data ######################
data_path = sample.data_path()

# Set parameters and read data
raw_fname = str(data_path) + "/MEG/sample/sample_audvis_filt-0-40_raw.fif"
event_fname = str(data_path) + "/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"
tmin, tmax = -0.0, 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method="iir")  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info["bads"] = ["MEG 2443"]  # set bad channels
picks = mne.pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
)

# Read epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=False,
    picks=picks,
    baseline=None,
    preload=True,
    verbose=False,
)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data() * 1000  # format is in (trials, channels, samples)
y = labels

# %%
kernels, chans, samples = 1, 60, 151

# take 50/25/25 percent of the data to train/validate/test
X_train = X[
    0:144,
]
Y_train = y[0:144]
X_validate = X[
    144:216,
]
Y_validate = y[144:216]
X_test = X[
    216:,
]
Y_test = y[216:]


# %%
# convert labels to one-hot encodings.
Y_train = np_utils.to_categorical(Y_train - 1)
Y_validate = np_utils.to_categorical(Y_validate - 1)
Y_test = np_utils.to_categorical(Y_test - 1)

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print("X_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")
