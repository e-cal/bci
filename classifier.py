import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as pre
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

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

if False:
    # This extracts the zip file
    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extract("data/mixed-data.csv", "extracted/")
    data = pd.read_csv("extracted/mixed-data.csv")

# ----- STEP 1: Load data -----
# Use os.path.join to construct the full path to the CSV file
file_path = os.path.join(
    r"C:\Users\danie\Documents\GitHub\bci\data", "one-game.csv"
)

# Use pandas to read the CSV file
data = pd.read_csv(file_path, sep=",", header=0)


def get_eeg(data):
    eeg_data = data.drop(
    columns=[
        "sample",
        "packet",
        "accel1",
        "accel2",
        "accel3",
        "other1",
        "other2",
        "other3",
        "other4",
        "other5",
        "other6",
        "other7",
        "analog1",
        "analog2",
        "analog3",
        "timestamp",
        ],
    )
    return eeg_data
data = get_eeg(data)

# -----  Step 2: Partition the data -----
def partition(data, partition_size):

    # Determine the size of each partition
    num_partitions = int(data.shape[0] / partition_size)

    # Split the DataFrame into partitions
    partitions = [
        data.iloc[i : i + partition_size, :]
        for i in range(0, data.shape[0], partition_size)
    ]
    for i in range(num_partitions):
        partition = partitions[i]
        active = partition['marker'].isin([1]).any()
        if active:
            partition['marker'] = 1
        if not active:
            partition['marker'] = 0

    return partitions, num_partitions

partition_size = 64
partitions, num_partitions = partition(data, partition_size)

# ----- Steps 3 and 4: Apply band pass filter and calculate band power -----
def power(partitions, num_partitions):
    # specify your desired band to calculate power in
    bands = {
        "Alpha": (8, 12),
        "Beta": (12, 30),
    }

    fs = 250  # sampling frequency

    band_data = pd.DataFrame(columns=range(16))
    print("Parition Count:", num_partitions)

    for i in range(num_partitions):
        # Get the data for the current partition
        partition = partitions[i]
        powers = []
        eeg_columns = partition.columns.drop('marker')

        for col in eeg_columns:
            filtered_data = pre.bandpass(
                partition[col], 8, 30, fs
            )  # bandpass filter from preprocessing file

            # Use Welch's method to estimate the power spectral density
            f, psd = signal.welch(filtered_data, fs, nperseg=64)

            for band in bands:
                # Find the indices of the frequency band
                idx_band = np.logical_and(f >= bands[band][0], f < bands[band][1])
                
                # Calculate the power in the frequency band
                power = np.trapz(psd[idx_band], f[idx_band])
                
                # Append the power to list
                powers.append(power)
        powers.append(partition['marker'][i*64])
        band_data = band_data.append(pd.Series(powers, index=range(17)), ignore_index=True)
    band_data = band_data.drop([0])
    return band_data

band_data = power(partitions, num_partitions)

# ----- Step 5: K=2 cluster -----

# Convert DataFrame to a numpy array
X = band_data.values # MAKE SURE TO DROP THE LABELS
print(X)

# Initialize PCA with 2 components
#pca = PCA(n_components=2)

# Fit the PCA model to the data and transform the data
#X_pca = pca.fit_transform(X)

# Update the DataFrame with the transformed data
df_pca = pd.DataFrame(X, columns=list(range(16)))

# Initialize KMeans model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Fit the model to the data
kmeans.fit(df_pca.values)

# Predict the cluster labels for each data point
guesses = kmeans.predict(df_pca)

# Add the cluster labels as a new column to the DataFrame
df_pca["cluster"] = guesses


# View the updated DataFrame
# Create a scatter plot with different colors for each cluster
#sns.scatterplot(x='PC1', y='PC2', hue=labels, data=df_pca)
plt.plot(df_pca["cluster"])
# Add labels to the plot
#plt.xlabel('PC1')
#plt.ylabel('PC2')

# Show the plot
plt.show()
