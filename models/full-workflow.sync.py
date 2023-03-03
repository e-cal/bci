# %% [markdown]
# # Imports

# %%
import logging
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from brainflow import *
from keras import layers
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split

# %% [markdown]
# Training Data

# %%
raw_training = DataFilter.read_file("../data/training_EEG.csv")
raw_training2 = DataFilter.read_file("../data/training-long_EEG.csv")
# raw_training3 = DataFilter.read_file("../data/combo_file.csv")

# %%
# raw_training = DataFilter.read_file('drive/MyDrive/eeg/training_EEG.csv')
# raw_training = pd.DataFrame(np.transpose(raw_training))
# raw_training = raw_training3

training_raw_channels = []
for channel in range(1, 9):
    training_raw_channels.append(raw_training[channel][:])
training_raw_channels = np.array(training_raw_channels)

training_raw_times = raw_training[22][:]
training_raw_markers = raw_training[23][:]

# %%
raw_training = DataFilter.read_file("../data/training-long_EEG.csv")
raw_training = pd.DataFrame(np.transpose(raw_training))

training_raw_channels = []
for channel in range(1, 9):
    training_raw_channels.append(raw_training[channel][:])
training_raw_channels = np.array(training_raw_channels)

training_raw_times = raw_training[22][:]
training_raw_markers = raw_training[23][:]

# %% [markdown]
# Testing Data

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
# #Trim and Scale Data
# Define scaling factor

# %%
SCALE_FACTOR = (4500000) / 24 / (2**23 - 1)

# %% [markdown]
# Remove the first and last 5 seconds of data

# %%
# Using 250 samples per second
fs = 250

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
# # Define Filters
# Notch Filters removes background power noise at 60 hz

# %%
def notch_filter(signal_data, notch_freq=60, notch_size=3, fs=250):
    notch_freq_Hz = np.array([float(notch_freq)])
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + notch_size * np.array([-1, 1])
        b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), "bandstop")
        fin = signal_data = signal.lfilter(b, a, signal_data)
    return fin


# %% [markdown]
# Bandpass filter smooths reduces gain depending on specified frequency band

# %%
def bandpass(start, stop, signal_data, fs=250):
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype="bandpass")
    return signal.lfilter(b, a, signal_data, axis=0)


# %% [markdown]
# #Apply Filters
#

# %%
filtered_training = []
filtered_testing = []

for i in range(8):
    # Filter training data
    notched = notch_filter(training_channels[i].T, notch_size=8)
    filtered_training.append(notched)

    # Filter testing data
    notched = notch_filter(testing_channels[i].T, notch_size=8)
    filtered_testing.append(notched)

# %% [markdown]
# #Fourier Transforms
# First we take the fourier transform of the entire dataset, averaging over all channels

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

# %% [markdown]
# Plot FFT of Filtered Training Data

# %% colab={"base_uri": "https://localhost:8080/", "height": 281}
plt.plot(training_freqs, training_avg_fourier)
plt.xlim(0, 125)
plt.ylim(0, 3000)
plt.title("FFT of Trainng Data")
plt.show()

# %% [markdown]
# Plot FFT of Filtered Testing Data

# %% colab={"base_uri": "https://localhost:8080/", "height": 281}
plt.plot(testing_freqs, testing_avg_fourier)
plt.xlim(0, 125)
plt.ylim(0, 3000)
plt.title("FFT of Testing Data")
plt.show()

# %% [markdown]
# Define paramaters for *(x_train, y_train)* generation
#

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
freq_lowerbound = 10
freq_upperbound = 100


# %% [markdown]
# Define function to generate *(x_   , y_    )* from the given paramaters

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


# %% [markdown]
# Get training and validation data

# %%
x_data, y_data, freq_list = get_processed_data(
    training_times, training_channels, training_spaces
)
x_valid, y_valid, freq_list = get_processed_data(
    testing_times, testing_channels, testing_spaces
)

# %% colab={"base_uri": "https://localhost:8080/"}
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

# %%
removal_ratio = 99 / 100

# %% colab={"base_uri": "https://localhost:8080/"}
#### DUMB SHIT TESTING
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

# %% colab={"base_uri": "https://localhost:8080/"}
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

# %%
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# %% colab={"base_uri": "https://localhost:8080/"}
print(f"x_train shape: {np.shape(x_train)}, y_train shape: {np.shape(y_train)}")
print(f"x_test shape: {np.shape(x_test)}, y_test shape: {np.shape(y_test)}")
print(
    f"Minimum training accuracy = {round(1 - np.sum(y_train) / np.shape(x_train)[0], 4)}"
)
print(
    f"Minimum validation accuracy = {round(1 - np.sum(y_test) / np.shape(x_test)[0], 4)}"
)


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

# %% [markdown]
# CNN

# %%
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

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

# %% colab={"base_uri": "https://localhost:8080/"}
print(np.shape(x_train))

# %% colab={"base_uri": "https://localhost:8080/"}
model.fit(x_train, y_train, batch_size=64, epochs=25, validation_data=(x_test, y_test))

# %% colab={"base_uri": "https://localhost:8080/", "height": 283}
plt.plot(model.history.history["val_accuracy"])

# %% [markdown]
# # Graphical Testing

# %%
y_pred = model(x_data)

# %% colab={"base_uri": "https://localhost:8080/"}
print(np.unique(y_pred))

# %% colab={"base_uri": "https://localhost:8080/", "height": 265}
start = 77450
interval = 1900

scaled = np.array(y_pred[start : start + interval])

plt.plot(y_data[start : start + interval])
# plt.plot(y_pred[start:start+interval])
plt.plot(scaled)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 372}
y_smoother = []
delta = 5
theshold = 0.75

y_smoother = [
    int(max(y_pred[i : i + delta]) > theshold) for i in range(len(y_pred) - delta)
]

for i in range(delta):
    y_smoother.append(0)


y_smoother = np.arrray(y_smoother)

# %%
start = 77450
interval = 1900

plt.plot(y_data[start : start + interval])
# plt.plot(y_pred[start:start+interval])
plt.plot(y_smoother[start : start + interval])
plt.show()

# %%
from matplotlib import rc

rc("animation", html="jshtml")
from math import *

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/", "height": 596}
delta_i = 800

n = 1000 - delta_i
samplerate = 80
fr = n // samplerate  # Number of frames
sim_time = 100  # time for sim to fully run
inter = n / (sim_time * samplerate)

fig, axs = plt.subplots(1, 1, figsize=(13, 10))

# Line Initialization
(line1,) = axs.plot([], [], lw=2)
(line2,) = axs.plot([], [], lw=2)


# Titles
# axs.set_title('Wavefunction Over Time', size = fs)

# fig.text(0.5, 0.95, "V(x) = {},".format(V(x)), ha='center', fontsize = fs)
# # trans = axs[0,1].get_xaxis_transform() # x in data untis, y in axes fraction
# axs.text(-0.35,0, "Frame =", ha="center", fontsize = fs)
# # time = axs[0,1].annotate('', xy=(1.1, 1.1), annotation_clip = False)
# time_temp = '%.1f'
# time = axs[0,0].text(0.55, 0.5, '', transform=axs[0,0].transAxes)

# #Axes Initialization
axs.set_ylim(-0.5, 1.5)
axs.set_xlim(0, 800)

# axs.set_xlabel('$x$ $pos$ ($arb. units$)')     # add labels
# axs.set_ylabel('$\psi$ ($arb. units$)')
axs.grid()


# Initialization function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return (
        line1,
        line2,
    )


# Animation function
def animate(i):
    # time.set_text(time_temp%(i))

    line1.set_data([j for j in range(delta_i)], y_data[i : i + delta_i])
    line2.set_data([j for j in range(delta_i)], y_pred[i : i + delta_i])
    # time.set_text(i*samplerate)
    return (
        line1,
        line2,
    )


# Call Animation
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=fr, interval=inter, blit=True
)
# plt.savefig('./h=0.02.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/", "height": 822}
anim

# %% colab={"base_uri": "https://localhost:8080/", "height": 283}
plt.plot(freq_list[:], x_test[59])

# %% [markdown]
# Testing with validation data

# %% colab={"background_save": true}
x_valid_copy = x_valid
y_valid_copy = y_valid

# %%
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)


x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)
y_valid = y_valid.reshape(y_valid.shape[0], 1)

# %% colab={"base_uri": "https://localhost:8080/"}
valid_pred = model(x_valid)
print(np.unique(valid_pred))

# %% colab={"base_uri": "https://localhost:8080/", "height": 266}
start = 9970
interval = 1600

scaled = 2 * np.array(valid_pred[start : start + interval])

plt.plot(y_valid[start : start + interval])
# plt.plot(y_pred[start:start+interval])
plt.plot(scaled)
plt.show()

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
# Prayer

from sklearn.ensemble import RandomForestClassifier
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# %% colab={"base_uri": "https://localhost:8080/", "height": 380}
# Option 1: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

# %%
# Option 2: Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

# %%
# Option 3: SVM
svm = SVC()
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)

# %%
# Option 5: Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred_nb = nb.predict(x_test)

# %%
print(np.unique(y_pred_nb))
print(nb.score(x_test, y_test))

# %%
start = 0
interval = 24000

scaled = np.array(y_pred_nb[start : start + interval])

# plt.plot(y_test[start:start+interval])
# plt.plot(y_pred[start:start+interval])
plt.plot(scaled)
plt.show()

# %% [markdown]
# #Clear GPU

# %%
tf.keras.backend.clear_session()
