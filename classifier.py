import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as pre
from scipy import signal

"""
--------
 This attempts to identify the jumping signals in Ethan's data jump.csv,
in which he repeatedly pressed spacebar on his computer
--------

The steps are as follows
1. Load data
2. Partition the data into small sections, hopefully some will contain the keystroke
3. Apply band pass filters to the data
4. Calculate the power of the motor band (12 - 30 hz)
5. Apply K=2 means clustering to the data, (jumping or not!)
"""


"""
# This extracts the zip file
with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extract("data/jump.csv", "extracted/")
data = pd.read_csv("extracted/jump.csv")
"""

# ----- STEP 1: Load data -----
# Use os.path.join to construct the full path to the CSV file
file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\bci\data\data", "jump.csv")

# Use pandas to read the CSV file
data = pd.read_csv(file_path, sep="\t", header=None)
data = data.drop([0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], axis=1)

# -----  Step 2: Partition the data -----
# Determine the number of partitions you want to create
num_partitions = 40

# Determine the size of each partition
partition_size = int(data.shape[0] / num_partitions)

# Split the DataFrame into partitions
partitions = [
    data.iloc[i : i + partition_size, :]
    for i in range(0, data.shape[0], partition_size)
]

# ----- Steps 3 and 4: Apply band pass filter and calculate band power -----

# specify your desired band to calculate power in
bands = {
    "Alpha": (8, 12),
    "Beta": (12, 30),
}  

fs = 250  # sampling frequency

# Initialize a dictionary to store the power in each frequency band
band_power = {band: [] for band in bands}

for i in range(1):
    # Get the data for the current partition
    partition = data[i * partition_size : (i + 1) * partition_size]
    # Apply FFT to each channel
    for col in partition.columns:
        filtered_data = pre.bandpass(
            partition[col], 12, 30, fs
        )  # banpass filter from preprocessing file

        # Use Welch's method to estimate the power spectral density
        f, psd = signal.welch(filtered_data, fs, nperseg=256)

        for band in bands:
            # Find the indices of the frequency band
            idx_band = np.logical_and(f >= bands[band][0], f < bands[band][1])
            # Calculate the power in the frequency band
            power = np.trapz(psd[idx_band], f[idx_band])
            # Append the power to the appropriate band in the dictionary
            band_power[band].append(power)

print(band_power['Beta'])
# -----
