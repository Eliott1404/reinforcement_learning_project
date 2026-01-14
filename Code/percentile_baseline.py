import numpy as np
import pandas as pd

def determine_percentiles(path="train.xlsx",boundaries=[0,0.4,0.6,1]):
    # If the first column contains the day/date labels:
    df = pd.read_excel(path, sheet_name=0)
    df = df.drop(columns=["PRICES"])

    percentiles = np.zeros((4, len(df.columns)))

    for i, hour in enumerate(df.columns):
        p0, lower, upper, p100 = df[hour].quantile(boundaries)
        percentiles[0, i] = lower
        percentiles[1, i] = lower - p0
        percentiles[2, i] = upper
        percentiles[3, i] = p100 - upper
    return percentiles

def percentile(observation, percentiles, proportion):
    price = observation[1]
    hour = int(observation[2]) - 1
    if price < percentiles[0][hour]:
        return proportion * (percentiles[0][hour] - price) / (percentiles[1][hour])
    elif price > percentiles[2][hour]:
        return proportion * (percentiles[0][hour] - price) / (percentiles[1][hour])
    else:
        return 0
    