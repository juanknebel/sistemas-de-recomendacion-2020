import numpy as np
import pandas as pd
import pickle
from scipy import spatial


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


def log(logger):
    def wrap(func):
        def wrapped_f(*args, **kwargs):
            logger.info(f"Entering {func.__name__} ...")
            result = func(*args, **kwargs)
            logger.info(f"Leaving {func.__name__} ...")
            return result

        return wrapped_f

    return wrap


def model_to_pickle(output):
    def wrap(func):
        def wrapped_f(*args, **kwargs):
            model = func(*args, **kwargs)
            pickle.dump(model, open(f"./data/06_models/model_{output}", "wb"))
            return model

        return wrapped_f

    return wrap


def cos_cdist(matrix, vector):
    """
    Compute the cosine distances between each row of matrix and vector.
    """
    v = vector.reshape(1, -1)
    return 1 - spatial.distance.cdist(matrix, v, "cosine").reshape(-1)
