import pandas as pd
import numpy as np


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
