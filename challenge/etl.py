import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


def expand_data_frame(df):
    df.insert(0, "user_id", range(len(df)))

    df_event_user = pd.DataFrame()
    print("Expand dataframe...")
    for row in tqdm(df.itertuples()):
        df_temp = pd.DataFrame(row.user_history)
        df_temp.insert(0, "user_id", [row.user_id] * len(df_temp))
        df_event_user = df_event_user.append(df_temp)

    return df_event_user.merge(
        df[["user_id", "item_bought"]], left_on="user_id", right_on="user_id"
    )


def save_to_csv(data, file_name):
    data.to_csv(file_name, index=False)


if __name__ == "__main__":
    pipeline_executions = {
        1: [expand_data_frame],
    }

    execution_help = ""
    for k, v in pipeline_executions.items():
        execution_help = (
            execution_help
            + f"Step {k} execute {', '.join([m.__name__ for m in v])}.\n"
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
        "-p",
        "--postfix",
        dest="postfix",
        type=int,
        nargs=None,
        help=f"number to indicate the postfix in the outputfiles",
        required=True,
    )
    args = parser.parse_args()

    if args.step:
        train = pd.read_json("./data/train_dataset.jl", lines=True)
        for a_method in pipeline_executions[args.step]:
            train, test = a_method(train)
            save_to_csv(train, f"./data/train_dataset{args.postfix}.csv")
