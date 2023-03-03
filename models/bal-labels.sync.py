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
from keras import constraints, layers
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# %% [markdown]
# # Load Data

# %%
data_0 = pd.read_csv("../data/60s-l-0.csv")
data_1 = pd.read_csv("../data/60s-arm-flap.csv")

SCALE_FACTOR = (4500000) / 24 / (2**23 - 1)
FREQ = 250  # 250 samples per second

# trim the first and last 5 seconds off the data
data_0 = data_0.iloc[5 * FREQ : -5 * FREQ, :]
data_1 = data_1.iloc[5 * FREQ : -5 * FREQ, :]

# scale the eeg columns (eeg1 - eeg8) by SCALE_FACTOR
eeg_cols = [f"eeg{i}" for i in range(1, 9)]
data_0[eeg_cols] = data_0[eeg_cols] * SCALE_FACTOR
data_1[eeg_cols] = data_1[eeg_cols] * SCALE_FACTOR

data = pd.concat([data_0, data_1])
data.head()

# %% [markdown]
# # Preprocess Data

# %%
eeg_data = []
for i in range(1, 9):
    eeg_data.append(data[f"eeg{i}"][:])
eeg_data = np.array(eeg_data)

timestamps = data["timestamp"][:]
markers = data["marker"][:]

# %% [markdown]
# ## Filters

# %%
NOTCH_SIZE = 8


def notch_filter(signal_data, notch_freq=60, notch_size=NOTCH_SIZE, fs=250):
    """Removes background noise at 60 hz (power lines)"""
    notch_freq_Hz = np.array([float(notch_freq)])
    fin = signal_data
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + notch_size * np.array([-1, 1])
        b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), "bandstop")
        fin = signal_data = signal.lfilter(b, a, signal_data)
    return fin


# %%
# def bandpass_filter(start, stop, signal_data, fs=250):
#     """Smooths and reduces gain depending on specified frequency band"""
#     bp_Hz = np.array([start, stop])
#     b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype="bandpass")
#     return signal.lfilter(b, a, signal_data, axis=0)


FREQ_LOW = 13
FREQ_HIGH = 80


def bandpass(data, lowcut=FREQ_LOW, highcut=FREQ_HIGH, fs=250):
    """Smooths and reduces gain depending on specified frequency band"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype="band")
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


# %%
# Apply filters
for col in eeg_cols:
    data_filtered = notch_filter(data[col])
    data_filtered = bandpass(data_filtered)
    data[col] = data_filtered

data.head()

# %% [markdown]
# ## Fourier Transforms

# %%
training_fourier = []

# Get frequency lists
training_freqs = np.fft.fftfreq(data["timestamp"].values.T.shape[-1], d=1 / FREQ)

# Get fourier transforms for each channel
for i in range(8):
    training_fourier.append(np.absolute(np.fft.fft(data[f"eeg{i+1}"].values.T)))

# Stack fourier transforms
training_stacked = np.stack(training_fourier, axis=1)

# Average fourier transforms
training_avg_fourier = np.mean(training_stacked, axis=1)

plt.plot(training_freqs, training_avg_fourier)
plt.xlim(0, 125)
plt.ylim(0, 1400)
plt.title("FFT of Trainng Data")
plt.show()

# %% [markdown]
# ## Split data

# %%
# Time window (in seconds) to consider preceding each space press
window_size = 0.05  # 50 ms

# Uncertainty (in seconds) around each space press marker
space_buffer = 0.02

# Latency assumption (in seconds)
latency = 0.02

# %%
def get_processed_data(
    times,
    channels,
    markers,
    window_size=window_size,
    space_buffer=space_buffer,
    latency=latency,
    n_width=NOTCH_SIZE,
    freq_lowerbound=FREQ_LOW,
    freq_upperbound=FREQ_HIGH,
):
    # Define return arrays
    x_data = []
    y_data = []

    fs = FREQ

    # The number of indices to consider preceding each space press
    delta_i = int(fs * window_size)

    space_i = int(fs * space_buffer)

    lat_i = int(fs * latency)

    # Number of trainingtime samples to consider
    num_samples = len(times) - delta_i - 1

    # Get frequency list
    freqs = np.fft.fftfreq(times[0:delta_i].shape[-1], d=1 / fs)

    # Record indices of chosen frequencies
    relevant_freq = []
    relevant_freq_indices = []

    for i, f in enumerate(freqs):
        if freq_lowerbound <= f <= freq_upperbound:
            relevant_freq.append(f)
            relevant_freq_indices.append(i)

    for i in range(num_samples):
        # Define window data
        window_data = []

        # Define window end index
        window_end = i + delta_i

        # Record space press
        y_data.append(max(markers[window_end - 2 - lat_i : window_end + 1 - lat_i]))

        # Get fourier transforms for each channel
        for channel in channels:
            fourier_series = np.absolute(np.fft.fft(channel[i : i + delta_i]))
            window_data.append(
                fourier_series[relevant_freq_indices[0] : relevant_freq_indices[-1] + 1]
            )

        # Stack fourier transforms
        window_stack = np.stack(window_data, axis=1)

        # Average fourier transforms
        window_avg = np.mean(window_stack, axis=1)

        x_data.append(np.array(window_avg))

        del window_data

    return np.array(x_data), np.array(y_data), relevant_freq


# %% [markdown]
# Get training and validation data

# %%
x_data, y_data, freq_list = get_processed_data(timestamps, eeg_data, markers)
x_data.shape, y_data.shape, freq_list

# %%
# Split data into training and validation sets
x_train: np.ndarray
x_val: np.ndarray
y_train: np.ndarray
y_val: np.ndarray
x_train, x_val, y_train, y_val = train_test_split(  # type: ignore
    x_data, y_data, test_size=0.2, shuffle=True, random_state=42
)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_train = x_train.reshape(1, 3, x_train.shape[0], 1)
# x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_val = x_val.reshape(1, 3, x_val.shape[0], 1)
# y_train = y_train.reshape(y_train.shape[0], 1)
y_train = y_train.reshape(1, 1, y_train.shape[0])
# y_val = y_val.reshape(y_val.shape[0], 1)
y_val = y_val.reshape(1, 1, y_val.shape[0])

x_train.shape, x_val.shape, y_train.shape, y_val.shape

# %%
# total = len(y_data)
# pos = np.sum(y_data)
# neg = total - pos

# class_weights = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
# class_weights

# %%
# np.sum(y_data), len(y_data)

# %% [markdown]
# # Models

# %% [markdown]
# ## CNN

# %%
model = tf.keras.Sequential(
    [
        layers.Conv2D(8, (1, FREQ // 2), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.DepthwiseConv2D(
            (3, 1),
            use_bias=False,
            depth_multiplier=2,
            depthwise_constraint=constraints.MaxNorm(1.0),
        ),
        layers.BatchNormalization(),
        layers.Activation("elu"),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(0.25),
        layers.SeparableConv2D(16, (1, 16), use_bias=False, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("elu"),
        layers.AveragePooling2D((1, 8)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(
            1, kernel_constraint=constraints.MaxNorm(0.25), activation="softmax"
        ),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# %%
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=32)

# %%
plt.plot(model.history.history["val_accuracy"])

# %%
y_pred = model(np.array(x_data))

start = 77450
interval = 1900

scaled = np.array(y_pred[start : start + interval])

plt.plot(y_data[start : start + interval])
# plt.plot(y_pred[start:start+interval])
plt.plot(scaled)
plt.show()
