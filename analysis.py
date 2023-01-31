import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

with pd.option_context("display.max_columns", None), pd.option_context("display.width", None), pd.option_context("display.max_colwidth", None):  # type: ignore
    p0 = df[df["packet"] == 0]["timestamp"].to_numpy()
    inc = []
    prev = 0
    for t in p0:
        inc.append(t > prev)
        prev = t
    print(all(inc))
