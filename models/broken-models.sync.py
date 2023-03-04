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

# %% [markdown]
# # Data Loading

# %% [markdown]
# ## Train


# %%
raw_training = DataFilter.read_file("../data/training_EEG.csv")


# %%
raw_training[1:9][0]

# %%
training_raw_channels = []
for channel in range(1, 9):
    training_raw_channels.append(raw_training[channel][:])
training_raw_channels = np.array(training_raw_channels)

training_raw_times = raw_training[22][:]
training_raw_markers = raw_training[23][:]

# %%
print(training_raw_channels.shape)

# %%
raw_training = DataFilter.read_file("../data/training_EEG.csv")
raw_training = pd.DataFrame(np.transpose(raw_training))

training_raw_channels = []
for channel in range(1, 9):
    training_raw_channels.append(raw_training[channel][:])
training_raw_channels = np.array(training_raw_channels)

training_raw_times = raw_training[22][:]
training_raw_markers = raw_training[23][:]

# %%
raw_training.head()

# %%
(training_raw_channels.T)[0]

# %% [markdown]
# ## Test

# %%
raw_testing = DataFilter.read_file("../data/testing_EEG.csv")
raw_testing = pd.DataFrame(np.transpose(raw_testing))

testing_raw_channels = []
for channel in range(1, 9):
    testing_raw_channels.append(raw_testing[channel][:])
testing_raw_channels = np.array(testing_raw_channels)

testing_raw_times = raw_testing[22][:]
testing_raw_markers = raw_testing[23][:]


# %% [markdown]
# # Data Prep

# %% [markdown]
# ## Trim and Scale Data

# %%
SCALE_FACTOR = (4500000) / 24 / (2**23 - 1)

# %%
# Using 250 samples per second
fs = 250

# trim and scale
training_times = np.array(training_raw_times[5 * fs : -5 * fs])
training_channels = np.array(
    [SCALE_FACTOR * training_raw_channels[n][5 * fs : -5 * fs] for n in range(8)]
)
training_spaces = np.array(training_raw_markers[5 * fs : -5 * fs])

testing_times = np.array(testing_raw_times[5 * fs : -5 * fs])
testing_channels = np.array(
    [SCALE_FACTOR * testing_raw_channels[n][5 * fs : -5 * fs] for n in range(8)]
)
testing_spaces = np.array(testing_raw_markers[5 * fs : -5 * fs])

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


# %%
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
filtered_training = []
filtered_testing = []

for i in range(8):
    # Filter training data
    notched = notch_filter(training_channels[i].T, notch_size=8)
    bpf = bandpass(notched)
    filtered_training.append(bpf)

    # Filter testing data
    notched = notch_filter(testing_channels[i].T, notch_size=8)
    bpf = bandpass(notched)
    filtered_testing.append(bpf)

# %% [markdown]
# ## Fourier Transforms

# %%
training_fourier = []
testing_fourier = []

# Get frequency lists
training_freqs = np.fft.fftfreq(training_times.shape[-1], d=1 / fs)
testing_freqs = np.fft.fftfreq(testing_times.shape[-1], d=1 / fs)

# Get fourier transforms for each channel
for i in range(8):
    training_fourier.append(np.absolute(np.fft.fft(filtered_training[i])))
    testing_fourier.append(np.absolute(np.fft.fft(filtered_testing[i])))

# Stack fourier transforms
training_stacked = np.stack(training_fourier, axis=1)
testing_stacked = np.stack(testing_fourier, axis=1)

# Average fourier transforms
training_avg_fourier = np.mean(training_stacked, axis=1)
testing_avg_fourier = np.mean(testing_stacked, axis=1)

# %%
plt.plot(training_freqs, training_avg_fourier)
plt.xlim(0, 125)
plt.ylim(0, 3000)
plt.title("FFT of Trainng Data")
plt.show()

# %%
plt.plot(testing_freqs, testing_avg_fourier)
plt.xlim(0, 125)
plt.ylim(0, 3000)
plt.title("FFT of Testing Data")
plt.show()

# %%
# Time window (in seconds) to consider preceding each space press
window_size = 0.15

# Uncertainty (in seconds) around each space press marker
space_buffer = 0.02

# Latency assumption (in seconds)
latency = 0.02

# Notch filter notch width
n_width = 6

# Range of frequencies to consider
freq_lowerbound = 13
freq_upperbound = 60


# %%
def get_processed_data(
    times,
    channels,
    markers,
    window_size=window_size,
    space_buffer=space_buffer,
    latency=latency,
    n_width=n_width,
    freq_lowerbound=freq_lowerbound,
    freq_upperbound=freq_upperbound,
):
    # Define return arrays
    x_data = []
    y_data = []

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

        # Record space press, 1 if
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
    return x_data, y_data, relevant_freq


# %%
x_data, y_data, freq_list = get_processed_data(
    training_times, training_channels, training_spaces
)
x_valid, y_valid, freq_list = get_processed_data(
    testing_times, testing_channels, testing_spaces
)

# %%
print(f"x_train shape: {np.shape(x_data)}, y_train shape: {np.shape(y_data)}")
print(f"x_valid shape: {np.shape(x_valid)}, y_valid shape: {np.shape(y_valid)}")
print(
    f"Minimum training accuracy = {round(1 - np.sum(y_data) / np.shape(x_data)[0], 4)}"
)
print(
    f"Minimum validation accuracy = {round(1 - np.sum(y_valid) / np.shape(x_valid)[0], 4)}"
)

# %% [markdown]
# # Partition data for training

# %%
# Reshape x_data to a 3D array
x_data = np.array(x_data)
y_data = np.array(y_data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, shuffle=True
)

# %% [markdown]
# # Removing 0 data

# %%
removal_ratio = 99 / 100

# %%
# Identify data points belonging to class 0
print(len(x_train))
class_1_indices = np.where(y_train == 0)[0]

# Randomly select half of the data points from class 1
num_points_to_remove = int(len(class_1_indices) * removal_ratio)
points_to_remove = np.random.choice(
    class_1_indices, size=num_points_to_remove, replace=False
)

# Remove the selected data points from the original dataset
x_train = np.delete(x_train, points_to_remove, axis=0)
y_train = np.delete(y_train, points_to_remove, axis=0)
print(len(x_train))

# %%
# Identify data points belonging to class 0
print(len(x_test))
class_1_indices = np.where(y_test == 0)[0]

# Randomly select half of the data points from class 1
num_points_to_remove = int(len(class_1_indices) * removal_ratio)
points_to_remove = np.random.choice(
    class_1_indices, size=num_points_to_remove, replace=False
)

# Remove the selected data points from the original dataset
x_test = np.delete(x_test, points_to_remove, axis=0)
y_test = np.delete(y_test, points_to_remove, axis=0)
print(len(x_test))

# %% [markdown]
# # Reshape data

# %%
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# %%
print(f"x_train shape: {np.shape(x_train)}, y_train shape: {np.shape(y_train)}")
print(f"x_test shape: {np.shape(x_test)}, y_test shape: {np.shape(y_test)}")
print(
    f"Minimum training accuracy = {round(1 - np.sum(y_train) / np.shape(x_train)[0], 4)}"
)
print(
    f"Minimum validation accuracy = {round(1 - np.sum(y_test) / np.shape(x_test)[0], 4)}"
)

# %%
x_train.shape


# %% [markdown]
# # Create Model
# Define custom loss function to penalize failuire to detect space presses more heavily
#

# %%
def custom_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    incorrect_pred = tf.abs(y_true - tf.round(y_pred))
    loss += tf.reduce_mean(
        tf.square(incorrect_pred * (1 + 100 * tf.abs(tf.cast(y_true, tf.float32))))
    )
    return loss


# %% [markdown]
# Define input shape

# %%
input_dim = (13,)


# %% [markdown]
# CNN

# %%
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# %%
x_train.shape


# %%
model = tf.keras.Sequential(
    [
        # layers.Reshape((input_dim, 1)),
        layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(units=1, activation="linear"),
    ]
)

# %% [markdown]
# Compile

# %%
# Compile the model

# Using mean squared error
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Using custom loss
model.compile(optimizer="adam", loss=custom_loss, metrics=["accuracy"])

# Print the model summary
# model.summary()

# %% [markdown]
# Train model

# %%
print(np.shape(x_train))


# %%
total = len(y_data)
pos = np.sum(y_data)
neg = total - pos

class_weights = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
class_weights

# %%
# set a valid path for your system to record model checkpoints
# checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
#                                save_best_only=True)

# fittedModel = model.fit(x_train, y_train, batch_size = 16, epochs = 300,
#                         verbose = 2, validation_data=(x_test, y_test),
#                         callbacks=[checkpointer], class_weight = class_weights)

# %%
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=25,
    validation_data=(x_test, y_test),
    class_weight=class_weights,
)

# %%
plt.plot(model.history.history["val_accuracy"])


# %% [markdown]
# # Graphical Testing

# %%
y_pred = model(np.array(x_data))

# %%
print(np.unique(y_pred))

# %%
start = 77450
interval = 1900

scaled = np.array(y_pred[start : start + interval])

plt.plot(y_data[start : start + interval])
# plt.plot(y_pred[start:start+interval])
plt.plot(scaled)
plt.show()

# %% [markdown]
# # Pytorch

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device.type

# %%
CHANNELS = 7
SAMPLE_HZ = 250


# %%
class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes,
        channels=CHANNELS,
        sample_hz=SAMPLE_HZ,
        dropout_rate=0.5,
        kernel_len=SAMPLE_HZ // 2,
        tfilter1=8,
        n_spatial_filters=2,
        tfilter2=None,  # tfilter1 * n_spatial_filters
        norm_rate=0.25,
        dropout_type="Dropout",
    ):
        super(EEGNet, self).__init__()
        self.nb_classes = n_classes
        self.channels = channels
        self.sample_hz = sample_hz
        self.dropout_rate = dropout_rate
        self.kernel_len = kernel_len
        self.tfilter1 = tfilter1
        self.n_spatial_filters = n_spatial_filters
        if tfilter2 is None:
            self.tfilter2 = tfilter1 * n_spatial_filters
        else:
            self.tfilter2 = tfilter2
        self.norm_rate = norm_rate

        if dropout_type == "Dropout2D":
            self.dropout = nn.Dropout2d
        elif dropout_type == "Dropout":
            self.dropout = nn.Dropout
        else:
            raise ValueError(
                "dropoutType must be one of Dropout2D or Dropout, passed as a string."
            )

        self.conv1 = nn.Conv2d(
            1,
            self.tfilter1,
            kernel_size=(1, self.kernel_len),
            padding=(0, self.kernel_len // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.tfilter1)
        self.depthwise = nn.Conv2d(
            self.tfilter1,
            self.tfilter1 * self.n_spatial_filters,
            kernel_size=(self.channels, 1),
            groups=self.tfilter1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.tfilter1 * self.n_spatial_filters)
        self.separable_conv = nn.Conv2d(
            self.tfilter1 * self.n_spatial_filters,
            self.tfilter2,
            kernel_size=(1, 16),
            padding=(0, 8),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.tfilter2)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self.tfilter2 * 4, self.nb_classes)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, kernel_size=(1, 4))
        x = self.dropout(p=self.dropout_rate)(x)

        x = self.separable_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, kernel_size=(1, 8))
        x = self.dropout(p=self.dropout_rate)(x)

        x = self.flatten(x)
        x = F.linear(
            x,
            self.dense.weight
            * torch.clamp(torch.norm(self.dense.weight), max=self.norm_rate)
            / torch.norm(self.dense.weight),
        )
        x = F.softmax(x, dim=1)

        return x


# %%
"""
# %%
# Reshape x_data to a 3D array
x_data = np.array(x_data)
y_data = np.array(y_data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, shuffle=True
)
"""
x_data_3d = np.array(x_data)
y_data_3d = np.array(y_data)
x_train, x_test, y_train, y_test = train_test_split(
    x_data_3d, y_data_3d, test_size=0.2, random_state=42, shuffle=False
)

# %%
# reshape for EEGNet
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

y_train = y_train.reshape(y_train.shape[0], 1)
y_train = y_train.reshape(y_train.shape[0], 1)

# %%
x_train.shape

# %%
# convert to torch tensor
x_train_tensor = torch.from_numpy(x_train).float().to(device)
x_test_tensor = torch.from_numpy(x_test).float().to(device)

y_train_tensor = torch.from_numpy(y_train).float().to(device)
y_test_tensor = torch.from_numpy(y_test).float().to(device)

# %%
# create dataset
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# %%
# create dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# %%
# Define the loss function and optimizer

# create model
model = EEGNet(n_classes=2).to(device)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Training loop

num_epochs = 10
for epoch in range(num_epochs):
    # Training
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for batch_idx, (df, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(df)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * df.size(0)
        _, pred = torch.max(outputs, 1)
        train_acc += torch.sum(pred == target.data)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)

    # Validation
    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (df, target) in enumerate(val_loader):
            outputs = model(df)
            loss = criterion(outputs, target)
            val_loss += loss.item() * df.size(0)
            _, pred = torch.max(outputs, 1)
            val_acc += torch.sum(pred == target.data)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_acc / len(val_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )


# %%

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, DepthwiseConv2D, Dropout, Flatten,
                          Input, MaxPooling2D, Permute, SeparableConv2D,
                          SpatialDropout2D)
from keras.models import Model
from keras.regularizers import l1_l2


def EEGNet(
    nb_classes,
    Chans=64,
    Samples=128,
    dropoutRate=0.5,
    kernLength=64,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout",
):
    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout
    else:
        raise ValueError(
            "dropoutType must be one of SpatialDropout2D "
            "or Dropout, passed as a string."
        )

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(Chans, Samples, 1),
        use_bias=False,
    )(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)

    return Model(inputs=input1, outputs=softmax)


# %%
model = EEGNet(
    nb_classes=2,
    Chans=7,
    Samples=250,
    dropoutRate=0.5,
    kernLength=250 // 2,
    F1=8,
    D=2,
    F2=16,
    dropoutType="Dropout",
)

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

# %%
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 250, 1)

# %%
total = len(y_data)
pos = np.sum(y_data)
neg = total - pos

class_weights = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
class_weights

# %%
# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(
    filepath="/tmp/checkpoint.h5", verbose=1, save_best_only=True
)

fittedModel = model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=300,
    verbose=2,
    validation_data=(x_test, y_test),
    callbacks=[checkpointer],
    class_weight=class_weights,
)

# %% [markdown]
# Reshape for RNN

# %%
# Reshape
x_valid = x_valid.reshape(1, x_valid.shape[0], x_valid.shape[1])
y_valid = y_valid.reshape(1, y_valid.shape[0])

# %%
# Reverse Reshape
x_valid = x_valid.reshape(x_valid.shape[1], x_valid.shape[2], 1)
y_valid = y_valid.reshape(y_valid.shape[1], 1)


# %% [markdown]
# Fully connected

# %%
# model = tf.keras.Sequential([
#     layers.Dense(1)
# ])
model = tf.keras.Sequential(
    [
        layers.Dense(16, activation="relu", input_shape=input_dim),
        layers.Dense(12, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1),
    ]
)

# %% [markdown]
# RNN

# %%
x_train = x_train.reshape(1, x_train.shape[0], x_train.shape[1])
x_test = x_test.reshape(1, x_test.shape[0], x_test.shape[1])

y_train = y_train.reshape(1, y_train.shape[0])
y_test = y_test.reshape(1, y_test.shape[0])

# %%
model = tf.keras.Sequential(
    [
        layers.InputLayer(input_shape=(None, input_dim[0])),
        layers.SimpleRNN(units=64, activation="sigmoid"),
        layers.Dense(units=1, activation="linear"),
    ]
)
