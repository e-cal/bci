import numpy as np
import pandas as pd

df = pd.read_csv("test1.csv")


LABEL_WINDOW_MS = 40

FREQ = 250  # sample rate (hz)
SAMPLE_GAP_MS = 1000 / FREQ  # time between samples
LABEL_WINDOW = np.ceil(LABEL_WINDOW_MS / SAMPLE_GAP_MS).astype(int)


def packets():
    with pd.option_context("display.max_columns", None), pd.option_context("display.width", None), pd.option_context("display.max_colwidth", None):  # type: ignore
        p0 = df[df["packet"] == 0]["timestamp"].to_numpy()
        inc = []
        prev = 0
        for t in p0:
            inc.append(t > prev)
            prev = t
        print(all(inc))


def markers():
    with pd.option_context("display.max_columns", None), pd.option_context("display.width", None), pd.option_context("display.max_colwidth", None):  # type: ignore
        marked = df[df["marker"] == 1]
        print(marked[["sample", "marker"]])
        print(len(marked) / LABEL_WINDOW)


markers()
