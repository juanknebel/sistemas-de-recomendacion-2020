#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:15:34 2020

@author: juan
"""


import pandas as pd
from tqdm import tqdm
import argparse
import time


def expand_data_frame(df):
    df.insert(0, "session_id", range(len(df)))

    event_user = {}
    for row in tqdm(df.itertuples()):
        event_user[row.session_id] = {
            "session_id": [row.session_id] * len(row.user_history),
            "event_info": list(
                map(lambda u: u["event_info"], row.user_history)
            ),
            "event_timestamp": list(
                map(lambda u: u["event_timestamp"], row.user_history)
            ),
            "event_type": list(
                map(lambda u: u["event_type"], row.user_history)
            ),
        }
    return pd.concat(
        {k: pd.DataFrame(v) for k, v in event_user.items()}, axis=0
    )


def expand_train(df):
    df_event_user = expand_data_frame(df)

    return df_event_user.merge(
        df[["session_id", "item_bought"]],
        left_on="session_id",
        right_on="session_id",
    )


def expand_test(df):
    return expand_data_frame(df)


def expand_items(df):
    df["site"] = list(map(lambda item: item[:3], df.category_id))
    df.condition = df[["condition"]].fillna("not_specified")
    df.domain_id = df[['domain_id']].fillna('unknown')
    # df.domain_id = list(
    #    map(lambda item: item[4:] if item != None else "unknown", df.domain_id)
    # )

    df.price = df[["price"]].fillna(-1)
    df.drop(columns=["product_id"], inplace=True)

    return df


def generate_random_sample(df, n=5000):
    sample = df.sample(n=n, random_state=int(time.time()))

    if "item_bought" in sample.columns:
        return expand_train(sample)
    else:
        return expand_test(sample)


def save_to_csv(data, file_name):
    data.to_csv(file_name, index=False)


if __name__ == "__main__":
    pipeline_executions = [
        expand_train,
        expand_test,
        expand_items,
        generate_random_sample,
    ]

    execution_help = (
        f"Methods: {', '.join([m.__name__ for m in pipeline_executions])}.\n"
    )

    parser = argparse.ArgumentParser(description="ETL cli.")
    parser.add_argument(
        "-s",
        "--step",
        dest="step",
        type=str,
        nargs=None,
        help=f"which method of the etl execute. {execution_help}",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=str,
        nargs=None,
        help="location of the file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        nargs=None,
        help="location to save the result",
        required=True,
    )
    args = parser.parse_args()

    if args.step:
        file_in = args.input
        file_out = args.output
        df = pd.read_json(file_in, lines=True)
        a_method = eval(args.step)
        df = a_method(df)
        save_to_csv(df, f"{file_out}")
