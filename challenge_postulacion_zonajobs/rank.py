import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import datetime
import multiprocessing
import csv
import logging
from src.utils import split_in, log

from sklearn.preprocessing import normalize
from src.utils import cos_cdist


logger = logging.getLogger("rank")
logger.setLevel(level=logging.INFO)

error_handler = logging.FileHandler("errors.log")
error_handler.setLevel(level=logging.ERROR)

debug_handler = logging.FileHandler("out.log")
debug_handler.setLevel(level=logging.DEBUG)

info_handler = logging.StreamHandler()
info_handler.setLevel(level=logging.INFO)

format_log = logging.Formatter(
    "%(asctime)s  %(name)s: %(levelname)s: %(message)s"
)
error_handler.setFormatter(format_log)
debug_handler.setFormatter(format_log)
info_handler.setFormatter(format_log)

logger.addHandler(error_handler)
logger.addHandler(debug_handler)
logger.addHandler(info_handler)


def rank(from_index: int, to_index: int, suffix):
    df_applicants_train = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_extended_train.csv"
    )
    df_applicants_to_predict = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_test.csv"
    )

    matrix_train = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_matrix_train.csv",
        index_col="idpostulante",
    )
    matrix_test = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_matrix_test.csv",
        index_col="idpostulante",
    )
    numpy_matrix_test = matrix_test.to_numpy()
    numpy_matrix_train = matrix_train.to_numpy()
    normalized_test = normalize(numpy_matrix_test, norm="l2")
    normalized_train = normalize(numpy_matrix_train, norm="l2")

    submission = pd.DataFrame()
    for index in tqdm(np.arange(from_index, to_index)):
        applicant_to_predict_vector = normalized_test[
            index
        ]  # matrix_test.iloc[index,].values
        applicant_to_predict_id = matrix_test.iloc[
            index,
        ].name
        treshold = 0.9
        top = 10
        cos = cos_cdist(normalized_train, applicant_to_predict_vector)
        sim = pd.DataFrame(
            {"idpostulante": matrix_train.index.values, "similaridad": cos}
        )
        top_applicants = (
            sim[(sim.similaridad > treshold)].head(top).idpostulante.values
        )

        notices_temp = (
            df_applicants_train[
                df_applicants_train.idpostulante.isin(top_applicants)
            ]
            .groupby("idaviso")
            .agg({"idpostulante": "count"})
            .reset_index()
        )
        notices_ids = notices_temp[
            notices_temp.idpostulante > 1
        ].idaviso.values[:10]
        notcies_len = len(notices_ids)
        applicant = df_applicants_to_predict[
            df_applicants_to_predict.idpostulante == applicant_to_predict_id
        ]

        to_rank = pd.DataFrame(
            {
                "idaviso": notices_ids,
                "idpostulante": [applicant.idpostulante.values[0]]
                * notcies_len,
            }
        )
        submission = pd.concat([submission, to_rank])

    submission.to_csv(
        f"./data/07_model_output/baseline_{suffix}.csv", index=False
    )


@log(logger)
def calculate_intervals(number_of_threads: int, total: int):
    interval = math.ceil(total / number_of_threads)
    intervals = []
    for i in range(number_of_threads):
        intervals += [(interval * i, min(interval * (i + 1), total))]
    return intervals


@log(logger)
def rank_similarity():
    total = 156232
    jobs = []
    number_of_threads = 8
    intervals = calculate_intervals(number_of_threads, total)
    for index, (f, t) in enumerate(intervals):
        p = multiprocessing.Process(
            target=rank,
            args=(
                f,
                t,
                index,
            ),
        )
        jobs.append(p)
        p.start()


@log(logger)
def rank_cold_start(
    test_attributes: pd.DataFrame,
    train_attributes: pd.DataFrame,
    df_applicants_train: pd.DataFrame,
    df_applicants_test: pd.DataFrame,
    top_ten_by_sex: dict,
    ranking_notice_by_applicant: pd.DataFrame,
    suffix: int,
):
    normalized_test = normalize(test_attributes.to_numpy(), norm="l2")
    normalized_train = normalize(train_attributes.to_numpy(), norm="l2")
    submission = {}
    treshold = 0.9
    top = 10

    for index in tqdm(range(len(test_attributes)), desc="Predicting"):
        applicant_to_predict_vector = normalized_test[
            index
        ]  # matrix_test.iloc[index,].values
        applicant_to_predict_id = test_attributes.iloc[
            index,
        ].name

        cos = cos_cdist(normalized_train, applicant_to_predict_vector)

        top_applicants = pd.DataFrame(
            {
                "idpostulante": train_attributes.index.values,
                "similaridad": cos,
            }
        )

        top_applicants = (
            top_applicants[top_applicants.similaridad > treshold]
            .head(top)
            .idpostulante.values
        )

        # notices_ids = ranking_notice_by_applicant[
        #    ranking_notice_by_applicant.idpostulante.isin(top_applicants)
        # ].idaviso.values[:10]

        notices_ids = (
            df_applicants_train[
                df_applicants_train.idpostulante.isin(top_applicants)
            ]
            .groupby("idaviso")
            .agg({"idpostulante": "count"})
            .sort_values(by="idpostulante", ascending=False)
            .index.values[:10]
        )

        # need at least 10 notices to predict
        notices_len = len(notices_ids)
        if notices_len < 10:
            sex = df_applicants_test[
                df_applicants_test.idpostulante == applicant_to_predict_id
            ].sexo.values[0]
            notices_ids = np.append(
                notices_ids, top_ten_by_sex[sex][: 10 - notices_len]
            )

        submission[applicant_to_predict_id] = notices_ids
    write_dict(submission, f"cold_start_{suffix}")


@log(logger)
def predict_cold_start(
    all_applicants: pd.DataFrame,
    applicants_test: pd.DataFrame,
    top_ten_by_sex: dict,
    ranking_notice_by_applicant: pd.DataFrame,
):
    matrix_train = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_matrix_train.csv",
        index_col="idpostulante",
    )

    matrix_test = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_matrix_test.csv",
        index_col="idpostulante",
    )

    matrix_test = matrix_test[
        matrix_test.index.isin(applicants_test.idpostulante)
    ]

    total = len(matrix_test)
    jobs = []
    number_of_threads = 8
    intervals = calculate_intervals(number_of_threads, total)
    for index, (f, t) in enumerate(intervals):
        filter_matrix_test = matrix_test.iloc[f:t]
        p = multiprocessing.Process(
            target=rank_cold_start,
            args=(
                filter_matrix_test,
                matrix_train,
                all_applicants,
                applicants_test,
                top_ten_by_sex,
                ranking_notice_by_applicant,
                index,
            ),
        )
        jobs.append(p)
        p.start()


def predict_hard_users(train: pd.DataFrame, test: pd.DataFrame):
    df_notice = pd.read_csv(
        "../data/02_intermediate/avisos/avisos_extended.csv"
    )

    rank_notices = (
        train.groupby("idaviso")
        .agg({"rank": "count"})
        .reset_index()
        .rename(columns={"rank": "cantidad"})
    )

    notice = (
        df_notice.merge(rank_notices, on="idaviso", how="left")
        .fillna(0)
        .drop(columns=["online_desde", "online_hasta"])
    )


@log(logger)
def write_dict(submission: dict, filename: str, header=None):
    with open(f"./data/07_model_output/{filename}.csv", mode="w") as out_file:
        out_writter = csv.writer(out_file, delimiter=",", quotechar="'")
        if header != None:
            out_writter.writerow(header)
        for applicant, notices in submission.items():
            out_writter.writerows(zip([applicant] * len(notices), notices))


@log(logger)
def rank_three_models():
    df_applicants_to_predict = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_test.csv"
    )
    dtypes = {"idaviso": "int64", "idpostulante": "string"}
    mydateparser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df_applicants_with_rank = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_train_rank.csv",
        parse_dates=["fechapostulacion"],
        date_parser=mydateparser,
        dtype=dtypes,
    )

    # Generate a notice top 10 rank by sex
    ranking_notice = (
        df_applicants_with_rank.groupby(["idaviso", "sexo"])
        .agg({"idpostulante": "count"})
        .reset_index()
        .rename(columns={"idpostulante": "cantidad"})
        .sort_values(by="cantidad", ascending=False)
    )

    top_ten_by_sex = {}
    top_ten_by_sex["FEM"] = (
        ranking_notice[ranking_notice.sexo == "FEM"].head(10).idaviso.values
    )
    top_ten_by_sex["MASC"] = (
        ranking_notice[ranking_notice.sexo == "MASC"].head(10).idaviso.values
    )
    top_ten_by_sex["NO_DECLARA"] = (
        ranking_notice[ranking_notice.sexo == "NO_DECLARA"]
        .head(10)
        .idaviso.values
    )

    # Generate a notice top rank by postulante
    ranking_notice_by_applicant = (
        df_applicants_with_rank.groupby(["idaviso", "idpostulante"])
        .agg({"rank": "count"})
        .reset_index()
        .rename(columns={"rank": "cantidad"})
        .sort_values(by="cantidad", ascending=False)
    )

    # Split the applicants to predict in two different groups
    # intersect: applicants that exist in both train and test
    # cold_start: applicants never seen in train
    ids_to_predict = set(df_applicants_to_predict.idpostulante)
    ids_train = set(df_applicants_with_rank.idpostulante)
    ids_cold_start = ids_to_predict - ids_train

    df_test_cold_start = df_applicants_to_predict[
        df_applicants_to_predict.idpostulante.isin(ids_cold_start)
    ]

    # Predict cold start
    predict_cold_start(
        df_applicants_with_rank,
        df_test_cold_start,
        top_ten_by_sex,
        ranking_notice_by_applicant,
    )

    # Split the the applicants that appear in train and test into two groups
    # hard_users: applicants that apply in more than 100 notices
    # low_users: applicants that apply in less than 100 notices
    intersect = ids_to_predict.intersection(ids_train)

    # Generate an applicant ranking
    ranking_by_applicant = (
        df_applicants_with_rank[
            df_applicants_with_rank.idpostulante.isin(intersect)
        ]
        .groupby("idpostulante")
        .agg({"idaviso": "count"})
        .reset_index()
        .rename(columns={"idaviso": "cantidad"})
        .sort_values(by="cantidad", ascending=False)
    )

    ids_hard_users = ranking_by_applicant[
        ranking_by_applicant.cantidad > 100
    ].idpostulante.values

    df_hard_users = df_applicants_with_rank[
        df_applicants_with_rank.idpostulante.isin(ids_hard_users)
    ]

    df_test_hard_users = df_applicants_to_predict[
        df_applicants_to_predict.idpostulante.isin(ids_hard_users)
    ]

    predict_hard_users(df_hard_users, df_test_hard_users)

    """
    ids_low_users = ranking_by_applicant[
        ranking_by_applicant.cantidad <= 100
    ].idpostulante.values

    df_low_users = df_applicants_with_rank[
        df_applicants_with_rank.idpostulante.isin(ids_low_users)
    ]

    df_test_low_users = df_applicants_to_predict[
        df_applicants_to_predict.idpostulante.isin(ids_low_users)
    ]
    """


if __name__ == "__main__":
    # rank_similarity()
    rank_three_models()
