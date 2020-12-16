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
import lightfm as lfm
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k


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


@log(logger)
def calculate_intervals(number_of_threads: int, total: int):
    interval = math.ceil(total / number_of_threads)
    intervals = []
    for i in range(number_of_threads):
        intervals += [(interval * i, min(interval * (i + 1), total))]
    return intervals


@log(logger)
def rank_cold_start(
    train_attributes: pd.DataFrame,
    test_attributes: pd.DataFrame,
    df_applicants_train: pd.DataFrame,
    df_applicants_test: pd.DataFrame,
    top_ten_by_sex: dict,
    suffix: int,
    return_dict: dict,
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
    filename = f"cold_start_{suffix}"
    write_dict(submission, filename)
    return_dict[f"thread_{suffix}"] = filename


@log(logger)
def predict_cold_start(
    applicants_with_live_notices: pd.DataFrame,
    applicants_test: pd.DataFrame,
    top_ten_by_sex: dict,
):
    matrix_train = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_matrix_train.csv",
        index_col="idpostulante",
    )

    matrix_train = matrix_train[
        matrix_train.index.isin(applicants_with_live_notices.idpostulante)
    ]

    matrix_test = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_matrix_test.csv",
        index_col="idpostulante",
    )

    matrix_test = matrix_test[
        matrix_test.index.isin(applicants_test.idpostulante)
    ]

    total = len(matrix_test)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    number_of_threads = 8
    intervals = calculate_intervals(number_of_threads, total)
    for index, (f, t) in enumerate(intervals):
        filter_matrix_test = matrix_test.iloc[f:t]
        p = multiprocessing.Process(
            target=rank_cold_start,
            args=(
                matrix_train,
                filter_matrix_test,
                applicants_with_live_notices,
                applicants_test,
                top_ten_by_sex,
                index,
                return_dict,
            ),
        )
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

    return [name for name in return_dict.values()]


def generate_user_feature(features, features_names):
    res = []
    for one_feature in features:
        one = []
        for index, feat_name in enumerate(features_names):
            one += [feat_name + ":" + str(one_feature[index])]
        res += [one]
    return res


@log(logger)
def generate_features(df_features):
    col = []
    value = []
    for a_column in df_features.columns.values:
        col += [a_column] * len(df_features[a_column].unique())
        value += list(df_features[a_column].unique())

    features = []
    for x, y in zip(col, value):
        res = str(x) + ":" + str(y)
        features += [res]
    return features


@log(logger)
def predict_hard_users(
    train: pd.DataFrame,
    test: pd.DataFrame,
    genre: pd.DataFrame,
    education: pd.DataFrame,
    notices: pd.DataFrame,
    available_notices: set,
):
    user_feature = genre.merge(education, on="idpostulante", how="left")
    user_feature["estudio"] = user_feature.nombre + "-" + user_feature.estado
    user_feature.drop(
        columns=["nombre", "estado", "fechanacimiento"], inplace=True
    )
    user_feature_hard_user = user_feature[
        user_feature.idpostulante.isin(train.idpostulante)
    ]

    uf = generate_features(user_feature[["sexo", "estudio"]])
    itf = generate_features(
        notices[
            ["nombre_zona", "tipo_de_trabajo", "nivel_laboral", "nombre_area"]
        ]
    )

    dataset1 = Dataset()
    dataset1.fit(
        train.idpostulante.unique(),  # all the users
        train.idaviso.unique(),  # all the items
        # user_features=uf,  # additional user features
    )
    # plugging in the interactions and their weights
    (interactions, weights) = dataset1.build_interactions(
        [(x[1], x[0], x[3]) for x in train.values]
    )

    feature_list = generate_user_feature(
        user_feature_hard_user[["sexo", "estudio"]].values, ["sexo", "estudio"]
    )
    #user_tuple = list(zip(user_feature_hard_user.idpostulante, feature_list))
    #user_features = dataset1.build_user_features(user_tuple, normalize=False)
    (
        user_id_map,
        user_feature_map,
        item_id_map,
        item_feature_map,
    ) = dataset1.mapping()
    inv_item_id_map = {v: k for k, v in item_id_map.items()}
    model = lfm.LightFM(loss="warp", random_state=42)
    model.fit(
        interactions,
        # user_features=user_features,
        sample_weight=weights,
        epochs=1000,
        num_threads=8,
    )

    test_precision = precision_at_k(
        # model, interactions, user_features=user_features, k=10, num_threads=8
        model,
        interactions,
        k=10,
        num_threads=8,
    ).mean()
    logger.info(f"Evaluation for LightFM is: {test_precision}")

    final_predictions = {}
    for a_user in tqdm(test.idpostulante.unique()):
        user_x = user_id_map[a_user]
        n_users, n_items = interactions.shape
        prediction = np.argsort(
            model.predict(
                # user_x, np.arange(n_items), user_features=user_features
                user_x,
                np.arange(n_items),
            )
        )[::-1]
        prediction_for_user = []
        for pred in prediction:
            notice = inv_item_id_map[pred]
            if notice in available_notices:
                prediction_for_user += [notice]
            if len(prediction_for_user) == 10:
                break
        final_predictions[a_user] = prediction_for_user

    write_dict(final_predictions, "lightfm")
    return ["lightfm"]


@log(logger)
def write_dict(submission: dict, filename: str, header=None):
    with open(f"./data/07_model_output/{filename}.csv", mode="w") as out_file:
        out_writter = csv.writer(out_file, delimiter=",", quotechar="'")
        if header != None:
            out_writter.writerow(header)
        for applicant, notices in submission.items():
            out_writter.writerows(zip(notices, [applicant] * len(notices)))


@log(logger)
def generate_top_ten_advice_by_sex(applicants: pd.DataFrame):
    # Generate a notice top 10 rank by sex
    ranking_notice = (
        applicants.groupby(["idaviso", "sexo"])
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

    return top_ten_by_sex


@log(logger)
def join_submission_files(
    final_submission: str, partial_submission_files: list
):
    all_submission = []
    for a_file in partial_submission_files:
        with open(f"./data/07_model_output/{a_file}.csv", mode="r") as in_file:
            in_reader = csv.reader(in_file, delimiter=",", quotechar="'")
            all_submission += list(in_reader)

    with open(
        f"./data/07_model_output/{final_submission}.csv", mode="w"
    ) as out_file:
        out_writter = csv.writer(out_file, delimiter=",", quotechar="'")
        out_writter.writerow(["idaviso", "idpostulante"])
        out_writter.writerows(all_submission)


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

    dtypes = {
        "idaviso": "int64",
        "tipo_de_trabajo": "string",
        "nivel_laboral": "string",
        "nombre_area": "string",
    }
    mydateparser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
    df_notice = pd.read_csv(
        "./data/02_intermediate/avisos/avisos_detalle.csv",
        parse_dates=["online_desde", "online_hasta"],
        date_parser=mydateparser,
        dtype=dtypes,
    )
    df_applicants_genre = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_genero_edad.csv"
    )
    df_applicants_education = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_educacion.csv"
    )

    # Me quedo con los notices que van a estar activos a partir de abril
    live_until = datetime.datetime(2018, 4, 1)
    notice_live_from = df_notice[df_notice.online_hasta >= live_until]
    applicants_sex_notices = df_applicants_with_rank.merge(
        notice_live_from[["idaviso"]], on="idaviso", how="inner"
    ).merge(
        df_applicants_genre[["idpostulante", "sexo"]],
        on="idpostulante",
        how="inner",
    )

    top_ten_by_sex = generate_top_ten_advice_by_sex(applicants_sex_notices)

    # Genero un diccionario con los usuarios y sus postulaciones
    applicants_notices_dict = {}
    for applicant, group in tqdm(
        df_applicants_with_rank.groupby("idpostulante")
    ):
        applicants_notices_dict[applicant] = set(group.idaviso.values)

    # Split the applicants to predict in two different groups
    # intersect: applicants that exist in both train and test
    # cold_start: applicants never seen in train
    ids_to_predict = set(df_applicants_to_predict.idpostulante)
    ids_train = set(df_applicants_with_rank.idpostulante)
    ids_cold_start = ids_to_predict - ids_train

    df_test_cold_start = df_applicants_to_predict[
        df_applicants_to_predict.idpostulante.isin(ids_cold_start)
    ].merge(df_applicants_genre, on="idpostulante", how="inner")

    submission_files = [
        "cold_start_0",
        "cold_start_1",
        "cold_start_2",
        "cold_start_3",
        "cold_start_4",
        "cold_start_5",
        "cold_start_6",
        "cold_start_7",
    ]
    # Predict cold start
    # submission_files += predict_cold_start(
    #    applicants_sex_notices,
    #    df_test_cold_start,
    #    top_ten_by_sex,
    # )

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

    # esto debe ser borrado! para hacer woosh
    ids_hard_users = intersect

    df_hard_users = df_applicants_with_rank[
        df_applicants_with_rank.idpostulante.isin(ids_hard_users)
    ]

    df_test_hard_users = df_applicants_to_predict[
        df_applicants_to_predict.idpostulante.isin(ids_hard_users)
    ]

    submission_files += predict_hard_users(
        df_hard_users,
        df_test_hard_users,
        df_applicants_genre,
        df_applicants_education,
        df_notice,
        set(notice_live_from.idaviso),
    )

    join_submission_files("final_submission", submission_files)

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
    import sys
    logger.info(f"Start experiment number: {sys.argv[1]}")
    rank_three_models()
