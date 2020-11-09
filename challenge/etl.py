#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:15:34 2020

@author: juan
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


def expand_data_frame(df, file_type):
    is_train = "train" == file_type
    is_test = "test" == file_type
    is_item = "item" == file_type

    if is_train or is_test:
        df.insert(0, "user_id", range(len(df)))

        event_user = {}
        for row in tqdm(df.itertuples()):
            event_user[row.user_id] = {
                "user_id": [row.user_id] * len(row.user_history),
                "event_info": list(map(lambda u: u["event_info"], row.user_history)),
                "event_timestamp": list(
                    map(lambda u: u["event_timestamp"], row.user_history)
                ),
                "event_type": list(map(lambda u: u["event_type"], row.user_history)),
            }
        df_event_user = pd.concat(
            {k: pd.DataFrame(v) for k, v in event_user.items()}, axis=0
        )

        if is_train:
            return df_event_user.merge(
                df[["user_id", "item_bought"]], left_on="user_id", right_on="user_id"
            )
        else:
            return df_event_user
    elif is_item:
        pass


def save_to_csv(data, file_name):
    data.to_csv(file_name, index=False)


if __name__ == "__main__":
    pipeline_executions = {
        1: [expand_data_frame],
    }

    execution_help = ""
    for k, v in pipeline_executions.items():
        execution_help = (
            execution_help + f"Step {k} execute {', '.join([m.__name__ for m in v])}.\n"
        )
    parser = argparse.ArgumentParser(description="ETL cli.")
    parser.add_argument(
        "-s",
        "--step",
        dest="step",
        type=int,
        nargs=None,
        help=f"which step of the etl execute. {execution_help}",
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        type=str,
        nargs=None,
        help="filename to preprocess",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="type",
        choices=["train", "test", "items"],
        help="type of file to preprocess",
    )
    parser.add_argument(
        "-p",
        "--postfix",
        dest="postfix",
        type=int,
        nargs=None,
        help="number to indicate the postfix in the outputfiles",
        required=True,
    )
    args = parser.parse_args()

    if args.step:
        file_name = args.file
        df = pd.read_json(file_name, lines=True)
        for a_method in pipeline_executions[args.step]:
            df = a_method(df, args.type)
            save_to_csv(df, f"{file_name}_{args.postfix}.csv")
