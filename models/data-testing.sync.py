# %% [markdown]
# # Imports

# %%
import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from brainflow import *
from keras import layers
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# %%
# raw_training = DataFilter.read_file("../data/training_EEG.csv")
# raw_training = pd.DataFrame(np.transpose(raw_training))
raw_training = pd.read_csv("../data/60s-l-0.csv")
raw_training.head()

# %%
SCALE_FACTOR = (4500000) / 24 / (2**23 - 1)
FREQ = 250  # 250 samples per second

# trim the first and last 5 seconds off the data
raw_training = raw_training.iloc[5 * FREQ : -5 * FREQ, :]

# scale the eeg columns (eeg1 - eeg8) by SCALE_FACTOR
eeg_cols = [f"eeg{i}" for i in range(1, 9)]
raw_training[eeg_cols] = raw_training[eeg_cols] * SCALE_FACTOR


# %%
training_raw_channels = []
for i in range(1, 9):
    training_raw_channels.append(raw_training[f"eeg{i}"][:])
training_raw_channels = np.array(training_raw_channels)

training_raw_times = raw_training["timestamp"][:]
training_raw_markers = raw_training["marker"][:]

# %%
raw_training.head()

# %%
(training_raw_channels.T)[0]

# %%
SCALE_FACTOR = (4500000) / 24 / (2**23 - 1)
fs = 250

# trim and scale
# training_times = np.array(training_raw_times[5 * fs : -5 * fs])
# training_channels = np.array(
#     [SCALE_FACTOR * training_raw_channels[n][5 * fs : -5 * fs] for n in range(8)]
# )
# training_spaces = np.array(training_raw_markers[5 * fs : -5 * fs])

# testing_times = np.array(testing_raw_times[5 * fs : -5 * fs])
# testing_channels = np.array(
#     [SCALE_FACTOR * testing_raw_channels[n][5 * fs : -5 * fs] for n in range(8)]
# )
# testing_spaces = np.array(testing_raw_markers[5 * fs : -5 * fs])

# %% [markdown]
# ## Filters

# %%
def notch_filter(signal_data, notch_freq=60, notch_size=3, fs=250):
    notch_freq_Hz = np.array([float(notch_freq)])
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + notch_size * np.array([-1, 1])
        b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), "bandstop")
        fin = signal_data = signal.lfilter(b, a, signal_data)
    return fin


def bandpass(start, stop, signal_data, fs=250):
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype="bandpass")
    return signal.lfilter(b, a, signal_data, axis=0)


filtered_training = []
filtered_testing = []

for i in range(8):
    # Filter training data
    notched = notch_filter(training_channels[i].T, notch_size=8)
    bpf = bandpass(1, 50, notched)
    filtered_training.append(notched)

    # Filter testing data
    # notched = notch_filter(testing_channels[i].T, notch_size=8)
    # filtered_testing.append(notched)

# %% [markdown]
# ## Fourier Transforms
# First we take the fourier transform of the entire dataset, averaging over all channels

# %%
training_fourier = []
# testing_fourier = []

# Get frequency lists
training_freqs = np.fft.fftfreq(training_times.shape[-1], d=1 / fs)
# testing_freqs = np.fft.fftfreq(testing_times.shape[-1], d=1 / fs)

# Get fourier transforms for each channel
for i in range(8):
    training_fourier.append(np.absolute(np.fft.fft(filtered_training[i])))
    # testing_fourier.append(np.absolute(np.fft.fft(filtered_testing[i])))

# Stack fourier transforms
training_stacked = np.stack(training_fourier, axis=1)
# testing_stacked = np.stack(testing_fourier, axis=1)

# Average fourier transforms
training_avg_fourier = np.mean(training_stacked, axis=1)
# testing_avg_fourier = np.mean(testing_stacked, axis=1)

# %% [markdown]
# Plot FFT of Filtered Training Data

# %%
plt.plot(training_freqs, training_avg_fourier)
plt.xlim(0, 125)
plt.ylim(0, 3000)
plt.title("FFT of Trainng Data")
plt.show()
