# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots

# %%

GAIN = 24
SCALE_FACTOR = (4500000) / 24 / (2**23 - 1)
FREQ = 250
WINDOW_SIZE = 60
ROLLING = True

BASE_SENS = 120
SENS = BASE_SENS * SCALE_FACTOR  # if SCALE_FACTOR != 1 else 150

# %%

eeg_cols = [f"eeg{i}" for i in range(1, 9)]


def get_eeg(data):
    data = data.drop(
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

    # trim the first and last 5 seconds off the data
    data = data.iloc[5 * FREQ :, :]

    # scale the eeg columns (eeg1 - eeg8) by SCALE_FACTOR
    data[eeg_cols] = data[eeg_cols] * SCALE_FACTOR
    return data


def reaction_adjust(df, n):
    """move labels n steps forward to account for reaction time"""
    labels = df["marker"].copy().values
    for i in range(len(labels) - n):
        if labels[i] == 1:
            labels[i] = 0
            labels[i + n] = 1
    df["marker"] = labels
    return df


def find_blinks(data):
    """Find blinks in EEG data"""
    window_size = WINDOW_SIZE
    for i in range(0, data.shape[0], window_size):
        if ROLLING:
            if (
                # np.mean(data["eeg1"].iloc[i : i + window_size])
                # - np.mean(data["eeg1"].iloc[(i - window_size) : i])
                data["eeg1"].iloc[i] - data["eeg1"].iloc[i - window_size]
                <= -SENS
            ):
                data["marker"][i] = 2

            elif (
                # np.mean(data["eeg1"].iloc[i : i + window_size])
                # - np.mean(data["eeg1"].iloc[(i - window_size) : i])
                data["eeg1"].iloc[i] - data["eeg1"].iloc[i - window_size]
                >= SENS - 50
            ):
                data["marker"][i] = 3
        else:
            if (
                np.mean(data["eeg1"].iloc[i : i + window_size])
                - np.mean(data["eeg1"].iloc[(i - window_size) : i])
                <= -SENS
            ):
                data["marker"][i] = 2

            elif (
                np.mean(data["eeg1"].iloc[i : i + window_size])
                - np.mean(data["eeg1"].iloc[(i - window_size) : i])
                >= SENS - 50
            ):
                data["marker"][i] = 3

    return data


# %%
data = pd.read_csv("../data/blink.csv", sep=",", header=0)
data = get_eeg(data)

if ROLLING:
    data2 = data[eeg_cols].rolling(WINDOW_SIZE).mean()
    data2["marker"] = data["marker"].iloc[60:]
    data = data2

data.head()

# %%
data = find_blinks(data)
# data[data["marker"] == 2]
n_windows = len(data[data["marker"] == 2])
print(f"Found {n_windows} windows")


# %%
rows = n_windows // 2 + 1
cols = 2
fig = make_subplots(rows, cols)

buffer = 150
pos = 0
for i in range(data.shape[0]):
    try:
        if data["marker"][i] == 2:
            row = pos // cols + 1
            col = pos % cols + 1

            fig.add_trace(
                go.Scatter(
                    x=list(range(i - buffer, i + buffer)),
                    y=data["eeg1"][i - buffer : i + buffer],
                ),
                row=row,
                col=col,
            )

            if data["marker"][i] == 2:
                fig.add_vline(x=i, row=row, col=col)
                for j in range(i, i + 150):
                    if data["marker"][j] == 3:
                        fig.add_vline(x=j, row=row, col=col)
                        break

            pos += 1
    except:
        pass

fig.show()
