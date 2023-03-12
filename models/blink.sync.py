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
STEP_SIZE = 60
ROLLING = False
WINDOW_SIZE = STEP_SIZE * 2
BLINK_LEN = 200

BASE_SENS = 150
SENS = BASE_SENS * SCALE_FACTOR  # if SCALE_FACTOR != 1 else 150

# %%

eeg_cols = [f"eeg{i}" for i in range(1, 9)]


def get_eeg(data):
    data = data.drop(
        columns=[
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
    start_point = 0
    for i in range(0, data.shape[0], STEP_SIZE):
        if ROLLING:
            check_start = (
                data["eeg1"].iloc[i] - data["eeg1"].iloc[i - STEP_SIZE] <= -SENS
            )
            check_end = data["eeg1"].iloc[i] - data["eeg1"].iloc[i - STEP_SIZE] >= SENS
        else:
            check_start = (
                np.mean(data["eeg1"].iloc[i : i + STEP_SIZE])
                - np.mean(data["eeg1"].iloc[(i - STEP_SIZE) : i])
                <= -SENS
            )
            check_end = (
                np.mean(data["eeg1"].iloc[i : i + STEP_SIZE])
                - np.mean(data["eeg1"].iloc[(i - STEP_SIZE) : i])
                >= SENS
            )

        if check_start:
            data["marker"][i] = 2
            start_point = BLINK_LEN

        elif check_end and start_point != 0:
            data["marker"][i] = 3
            start_point = 0

        if start_point > 0:
            start_point -= 1

    return data


# %%
data = pd.read_csv("../data/blink-3m.csv", sep=",", header=0)
data = get_eeg(data)

if ROLLING:
    data2 = data[eeg_cols].rolling(STEP_SIZE).mean()
    data2["marker"] = data["marker"].iloc[60:]
    data = data2

data.head()

# %%
data = find_blinks(data)
# data[data["marker"] == 2]
n_windows = len(data[data["marker"] == 2])
print(f"Found {n_windows} windows")


# %%
max_rows = 4
rows = min(n_windows // 2 + 1, max_rows)
cols = 2
fig = make_subplots(rows, cols)

pos = 0
for i in range(data.shape[0]):
    try:
        if data["marker"][i] == 1:
            row = pos // cols + 1
            col = pos % cols + 1

            fig.add_trace(
                go.Scatter(
                    x=list(range(i - WINDOW_SIZE, i + BLINK_LEN)),
                    y=data["eeg1"][i - WINDOW_SIZE : i + BLINK_LEN],
                ),
                row=row,
                col=col,
            )

            # if data["marker"][j] == 1:
            fig.add_vline(x=i, row=row, col=col)

            for j in range(i - WINDOW_SIZE, i + BLINK_LEN):
                if data["marker"][i] == 2:
                    fig.add_vline(x=i, row=row, col=col, line_dash="dash")
                if data["marker"][j] == 3:
                    fig.add_vline(x=j, row=row, col=col, line_dash="dot")

            pos += 1
    except:
        pass
    if pos > rows * cols:
        break

fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(data.shape[0])), y=data["eeg1"]))

for i in range(data.shape[0]):
    if data["marker"].iloc[i] == 1:
        fig.add_vline(x=i)
    if data["marker"].iloc[i] == 2:
        fig.add_vline(x=i, line_dash="dash")
    if data["marker"].iloc[i] == 3:
        fig.add_vline(x=i, line_dash="dot")

fig.show()

# %%
n_blinks = len(data[data["marker"] == 1])
n_start = len(data[data["marker"] == 2])
n_end = len(data[data["marker"] == 3])

print(f"{n_blinks} real blinks")
print(f"{n_start} start blinks")
print(f"{n_end} end blinks")

# %%
blinked = False
blink_start = 0
no_blink = 0
pred_start, pred_end = False, False
fn = 0
fp = 0
tp = 0
tn = 0
for i in range(data.shape[0]):
    if data["marker"].iloc[i] == 1:
        blinked = True
        blink_start = BLINK_LEN
        no_blink = 0

    if data["marker"].iloc[i] == 2:
        pred_start = True

    if data["marker"].iloc[i] == 3:
        pred_end = True

    if (
        blinked
        and blink_start == 0
        and (not pred_start or (pred_start and not pred_end))
    ):
        fn += 1
        blinked = False
        pred_start, pred_end = False, False

    if not blinked and pred_start:
        fp += 1
        pred_start, pred_end = False, False

    if blinked and pred_start and pred_end:
        tp += 1
        blinked = False
        pred_start, pred_end = False, False

    if blink_start > 0:
        blink_start -= 1

    if not blinked:
        no_blink += 1

    if no_blink > BLINK_LEN and not pred_start:
        tn += 1
        no_blink = 0


print(f"TP: {tp}")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")

# %%
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1: {f1*100:.2f}%")

# %%
blink_locs = [i for i in range(data.shape[0]) if data["marker"].iloc[i] == 1]

# %%
data0 = []
data1 = []
for i in blink_locs:
    data1.append(data.iloc[i - WINDOW_SIZE : i + BLINK_LEN].values)
    print(len(data1[0]))
    data0.append(data.iloc[i + BLINK_LEN : i + WINDOW_SIZE + BLINK_LEN].values)
    print(len(data0[0]))
    break
