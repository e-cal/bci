import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as pre
from scipy import signal
import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
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
    r"C:\Users\danie\Documents\GitHub\bci\data", "3-games.csv"
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

#grab the data
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
        partition = pd.DataFrame(partitions[i])

        active = np.isin(1, partition['marker'].tolist())

        if active:
            partition['marker'][:] = 1
        if not active:
            partition['marker'][:] = 0

    return partitions, num_partitions

partition_size = 256
partitions, num_partitions = partition(data, partition_size)
print(num_partitions)
# ----- Steps 3 and 4: Apply band pass filter and calculate band power -----
def power(partitions, num_partitions, partition_size):
    # specify your desired band to calculate power in
    bands = {
        "Beta": (12, 30),
    }

    fs = 250  # sampling frequency

    band_data = pd.DataFrame(columns=range(8))
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
            f, psd = signal.welch(filtered_data, fs, nperseg=partition_size/2)

            for band in bands:
                # Find the indices of the frequency band
                idx_band = np.logical_and(f >= bands[band][0], f <= bands[band][1])

                # Calculate the power in the frequency band
                power = np.trapz(psd[idx_band], f[idx_band])
                
                # Append the power to list
                powers.append(power)

        powers.append(partition['marker'][i*partition_size+1])


        band_data = band_data.append(pd.Series(powers, index=range(9)), ignore_index=True)

    band_data = band_data.drop([0])
    return band_data

band_data = power(partitions, num_partitions, partition_size)
print(np.shape(band_data))
# ----- Step 5: K=2 cluster -----

train = pd.DataFrame(band_data.values, columns = list(range(9)))

X = np.array(train.drop(columns = 8))
y = np.array(train[8])

# 5 fold CV
n_folds = 10
cv_scores = []
for folds in range(4,10):
    print(f"folds: {folds}")
    # create a KFold object to split the data into n_folds folds
    kf = KFold(n_splits=folds, shuffle=True)
    fold_scores = []
    # iterate over the folds and train/validate the model
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        # split the data into train and validation sets
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        
        # create and train the model on the training set
        #model = KMeans(n_clusters=2,n_init=10)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # evaluate the model on the validation set
        y_pred = model.predict(X_val)

        score = sk.metrics.accuracy_score(y_val,y_pred)
        fold_scores.append(score)
        #print("Fold {}: Accuracy = {:.2f}".format(fold+1, score))
    cv_scores.append(np.mean(fold_scores))

plt.plot(range(4,10),cv_scores)
plt.xlabel("CV folds")
plt.ylabel("accuracy")
plt.show()