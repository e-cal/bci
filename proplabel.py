import numpy as np
import pandas as pd

# def propagate_label(df, n):
#     labels = df["label"].copy()
#     for i in range(len(df)):
#         if labels[i] == 1:
#             start = max(0, i - n)
#             labels[start:i] = 1
#     df["label"] = labels
#     return df


def propagate_label(df, n):
    labels = df["label"].copy().values
    for i in range(len(labels)):
        if labels[i] == 1:
            start = max(0, i - n)
            labels[start:i] = 1
    df["label"] = labels
    return df


def propagate_label_np(df, n):
    labels = df["label"].copy().values
    ones = np.where(labels == 1)[0]
    starts = np.maximum.accumulate(np.maximum(ones - n, 0))
    mask = np.zeros_like(labels)
    for i, start in enumerate(starts):
        mask[start : ones[i]] = 1
    labels[mask.astype(bool)] = 1
    df["label"] = labels
    return df


def test_propagate_label():
    df = pd.DataFrame(
        {"label": [0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
    )
    print(df)
    expected = pd.DataFrame(
        {"label": [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]},
    )
    result = propagate_label_np(df, 2)
    print(result)
    print("expect")
    print(expected)
    pd.testing.assert_frame_equal(result, expected)


def test_speed():
    import time

    df = pd.DataFrame(
        {"label": [0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
    )

    n = 1000
    total = 0
    for _ in range(n):
        t = time.time()
        propagate_label_np(df, 2)
        total += time.time() - t

    print(total / n)


# test_propagate_label()
test_speed()
