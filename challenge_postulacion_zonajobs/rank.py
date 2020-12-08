import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import multiprocessing

from sklearn.preprocessing import normalize
from src.utils import cos_cdist


def rank(from_index, to_index, suffix):
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


def calculate_intervals(number_of_threads, total):
    interval = math.ceil(total / number_of_threads)
    intervals = []
    for i in range(number_of_threads):
        intervals += [(interval * i, min(interval * (i + 1), total))]
    return intervals


if __name__ == "__main__":
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
