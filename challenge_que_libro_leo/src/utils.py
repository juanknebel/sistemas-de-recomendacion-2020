import numpy as np
import pandas as pd


def split_in(index_list, k):
    count = int(len(index_list) / k)
    index_split = {}
    for i in range(k - 1):
        split_i = np.random.choice(index_list, count, False)
        index_split[i] = split_i
        index_list = list(set(index_list) - set(split_i))
    index_split[i + 1] = index_list

    return index_split


def extended_describe(data_frame, decimals=2, include_types=[np.number]):
    describe = data_frame.describe(include=include_types).round(decimals)
    na_describe = (
        data_frame[describe.columns]
        .isnull()
        .sum()
        .to_frame(name="missing")
        .T.round(decimals)
    )
    return pd.concat([describe, na_describe])
