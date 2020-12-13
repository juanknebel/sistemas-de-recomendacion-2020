from src.utils import split_in, log
import pandas as pd
import numpy as np
import datetime
import logging
from tqdm import tqdm

logger = logging.getLogger("etl")
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
def remove_duplicates():
    df_applicants_education = pd.read_csv(
        "./data/01_raw/postulantes/postulantes_educacion.csv"
    )
    df_applicants_education.drop_duplicates(inplace=True)
    df_applicants_education.to_csv(
        "./data/02_intermediate/postulantes/postulantes_educacion.csv",
        index=False,
    )

    df_applicants_genre_year = pd.read_csv(
        "./data/01_raw/postulantes/postulantes_genero_edad.csv"
    )
    df_applicants_genre_year.drop_duplicates(
        subset=["idpostulante"], inplace=True
    )

    df_applicants_genre_year.to_csv(
        "./data/02_intermediate/postulantes/postulantes_genero_edad.csv",
        index=False,
    )


@log(logger)
def clean():
    df_applicants_genre_year = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_genero_edad.csv"
    )

    df_applicants_genre_year.loc[
        df_applicants_genre_year.sexo == "0.0", "sexo"
    ] = "NO_DECLARA"

    df_applicants_genre_year.to_csv(
        "./data/02_intermediate/postulantes/postulantes_genero_edad.csv",
        index=False,
    )

    dtypes = {
        "idaviso": "int64",
        "idpais": "int64",
        "titulo": "string",
        "descripcion": "string",
        "nombre_zona": "string",
        "ciudad": "str",
        "mapacalle": "string",
        "tipo_de_trabajo": "string",
        "nivel_laboral": "string",
        "nombre_area": "string",
        "denominacion_empresa": "string",
    }
    df_notice_detail = pd.read_csv(
        "./data/01_raw/avisos/avisos_detalle.csv", dtype=dtypes
    )

    dtypes = {"idaviso": "int64"}
    mydateparser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
    df_notice_online = pd.read_csv(
        "./data/01_raw/avisos/avisos_online.csv",
        parse_dates=["online_desde", "online_hasta"],
        date_parser=mydateparser,
        dtype=dtypes,
    )

    df_notice_online = df_notice_online.merge(df_notice_detail, on="idaviso")
    df_notice_online.drop(
        columns=[
            "idpais",
            "titulo",
            "descripcion",
            "nombre_zona",
            "ciudad",
            "mapacalle",
            "denominacion_empresa",
        ],
        inplace=True,
    )

    df_notice_online.to_csv(
        "./data/02_intermediate/avisos/avisos_extended.csv", index=False
    )


@log(logger)
def join_train():
    df_notice_detail = pd.read_csv(
        "./data/02_intermediate/avisos/avisos_extended.csv"
    )

    dtypes = {"idaviso": "int64", "idpostulante": "string"}
    mydateparser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df_applicants_train = pd.read_csv(
        "./data/01_raw/postulaciones/postulaciones_train.csv",
        parse_dates=["fechapostulacion"],
        date_parser=mydateparser,
        dtype=dtypes,
    )

    df_applicants_genre_year = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_genero_edad.csv",
        dtype=dtypes,
    )
    df_applicants_genre_year["fechanacimiento"] = pd.to_datetime(
        df_applicants_genre_year["fechanacimiento"], errors="coerce"
    )

    df = df_applicants_train.merge(
        df_notice_detail[
            ["idaviso", "tipo_de_trabajo", "nivel_laboral", "nombre_area"]
        ],
        on="idaviso",
    )
    df = df.merge(df_applicants_genre_year, on="idpostulante")
    df["anios_al_postularse"] = (
        df.fechapostulacion - df.fechanacimiento
    ).astype("timedelta64[Y]")
    df.drop(columns=["fechapostulacion", "fechanacimiento"], inplace=True)
    df["id"] = range(len(df))

    # Busco las edades promedio para completar los datos faltantes
    mean_age_group = (
        df.dropna()
        .groupby(["sexo", "nivel_laboral", "tipo_de_trabajo", "nombre_area"])
        .agg({"anios_al_postularse": np.mean})
        .reset_index()
    )
    years_na = df[df.anios_al_postularse.isna()].merge(
        mean_age_group,
        on=["sexo", "nivel_laboral", "tipo_de_trabajo", "nombre_area"],
        how="left",
    )
    df.loc[
        df.id.isin(years_na.id), "anios_al_postularse"
    ] = years_na.anios_al_postularse_y.values

    mean_age_group = (
        df.dropna()
        .groupby(["sexo", "nivel_laboral", "tipo_de_trabajo"])
        .agg({"anios_al_postularse": np.mean})
        .reset_index()
    )
    years_na = df[df.anios_al_postularse.isna()].merge(
        mean_age_group,
        on=["sexo", "nivel_laboral", "tipo_de_trabajo"],
        how="left",
    )
    df.loc[
        df.id.isin(years_na.id), "anios_al_postularse"
    ] = years_na.anios_al_postularse_y.values

    mean_age_group = (
        df.dropna()
        .groupby(["sexo", "nivel_laboral"])
        .agg({"anios_al_postularse": np.mean})
        .reset_index()
    )
    years_na = df[df.anios_al_postularse.isna()].merge(
        mean_age_group, on=["sexo", "nivel_laboral"], how="left"
    )
    df.loc[
        df.id.isin(years_na.id), "anios_al_postularse"
    ] = years_na.anios_al_postularse_y.values

    df.drop(columns="id", inplace=True)
    df.to_csv(
        "./data/02_intermediate/postulaciones/postulaciones_extended_train.csv",
        index=False,
    )


@log(logger)
def calculate_applicants_matrix():
    df_applicants_train = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_extended_train.csv"
    )

    df_applicants_to_predict = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_test.csv"
    )

    dtypes = {"idpostulante": "string", "nombre": "string", "estado": "string"}
    df_applicants_education = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_educacion.csv",
        dtype=dtypes,
    )

    dtypes = {
        "idpostulante": "string",
        "sexo": "string",
        "fechanacimiento": "string",
    }
    df_applicants_genre_year = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_genero_edad.csv",
        dtype=dtypes,
    )
    df_applicants_genre_year["fechanacimiento"] = pd.to_datetime(
        df_applicants_genre_year["fechanacimiento"], errors="coerce"
    )

    test = df_applicants_to_predict.merge(
        df_applicants_genre_year, on="idpostulante"
    ).merge(df_applicants_education, on="idpostulante", how="left")
    test["value"] = 1
    test.drop(columns="fechanacimiento", inplace=True)

    matrix_test = test.pivot(
        index="idpostulante",
        columns=["sexo", "nombre", "estado"],
        values="value",
    ).fillna(0)
    matrix_test.columns = matrix_test.columns.to_flat_index()

    train = df_applicants_train[["idpostulante"]].drop_duplicates()
    train = train.merge(df_applicants_genre_year, on="idpostulante").merge(
        df_applicants_education, on="idpostulante", how="left"
    )
    train["value"] = 1
    train.drop(columns="fechanacimiento", inplace=True)

    matrix_train = train.pivot(
        index="idpostulante",
        columns=["sexo", "nombre", "estado"],
        values="value",
    ).fillna(0)
    matrix_train.columns = matrix_train.columns.to_flat_index()

    """
    tuples_equals = (lambda x, y: x[0] == y[0] and x[1] == y[1] and x[2] == y[2])
    for train_tuple in set(matrix_train.columns.values):
        found = False
        for test_tuple in set(matrix_test.columns.values):
            if tuples_equals(train_tuple, test_tuple):
                found = True
        if not found:
            print(train_tuple)

    tuples_equals = (lambda x, y: x[0] == y[0] and x[1] == y[1] and x[2] == y[2])
    for test_tuple in set(matrix_test.columns.values):
        found = False
        for train_tuple in set(matrix_train.columns.values):
            if tuples_equals(test_tuple, train_tuple):
                found = True
        if not found:
            print(test_tuple)
    """

    columns_dict = {}
    for index, test_tuple in enumerate(set(matrix_test.columns.values)):
        columns_dict[test_tuple] = f"atr_{index:02}"

    matrix_train.rename(columns=columns_dict, inplace=True)
    matrix_train = matrix_train.reindex(sorted(matrix_train.columns), axis=1)

    matrix_test.rename(columns=columns_dict, inplace=True)
    matrix_test = matrix_test.reindex(sorted(matrix_test.columns), axis=1)

    matrix_train.to_csv(
        "./data/02_intermediate/postulantes/postulantes_matrix_train.csv"
    )
    matrix_test.to_csv(
        "./data/02_intermediate/postulantes/postulantes_matrix_test.csv"
    )

    keys = []
    values = []
    for k, v in columns_dict.items():
        keys += [v]
        values += [k]
    pd.DataFrame({"key": keys, "value": values}).to_csv(
        "../data/02_intermediate/postulantes/atributos_diccionario.csv",
        index=False,
    )

    """
    numpy_matrix_test = matrix_test.to_numpy()
    numpy_matrix_train = matrix_train.to_numpy()

    np.savetxt(
        f"./data/02_intermediate/postulantes/numpy_matrix_test.csv",
        numpy_matrix_test,
        delimiter=",",
    )
    np.savetxt(
        f"./data/02_intermediate/postulantes/numpy_matrix_train.csv",
        numpy_matrix_train,
        delimiter=",",
    )
    """


@log(logger)
def generate_test():
    df_applicants_to_predict = pd.read_csv(
        "./data/01_raw/ejemplo_de_solucion.csv"
    ).drop_duplicates(subset=["idpostulante"])

    df_applicants_genre_year = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_genero_edad.csv"
    )

    df_applicants_to_predict.drop(columns="idaviso", inplace=True)
    df_applicants_to_predict.drop_duplicates(
        subset=["idpostulante"], inplace=True
    )

    df_applicants_to_predict = df_applicants_to_predict.merge(
        df_applicants_genre_year, on="idpostulante", how="left"
    )

    df_applicants_to_predict.to_csv(
        "./data/02_intermediate/postulaciones/postulaciones_test.csv",
        index=False,
    )


@log(logger)
def add_rank():
    dtypes = {"idaviso": "int64", "idpostulante": "string"}
    mydateparser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df_applicants_raw = pd.read_csv(
        "./data/01_raw/postulaciones/postulaciones_train.csv",
        parse_dates=["fechapostulacion"],
        date_parser=mydateparser,
        dtype=dtypes,
    )

    applicants_with_rank = []

    for _, group in tqdm(df_applicants_raw.groupby("idpostulante")):
        temp = group.sort_values(by="fechapostulacion", ascending=False)
        temp["rank"] = range(len(temp))
        applicants_with_rank += [temp]

    df_applicants_with_rank = pd.concat(applicants_with_rank)

    extended = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_extended_train.csv"
    )
    extended = (
        extended.sort_values(by=["idpostulante", "idaviso"])
        .reset_index()
        .drop(columns="index")
    )
    df_applicants_with_rank = (
        df_applicants_with_rank.sort_values(by=["idpostulante", "idaviso"])
        .reset_index()
        .drop(columns="index")
    )

    extended["fechapostulacion"] = df_applicants_with_rank["fechapostulacion"]
    extended["rank"] = df_applicants_with_rank["rank"]
    extended["id_test"] = df_applicants_with_rank["idpostulante"]
    to_check = extended.id_test == extended.idpostulante

    if not np.all(to_check):
        raise Exception("Ids aren't the same")

    extended.sort_values(by=["idpostulante", "rank"], inplace=True)
    extended.drop(columns="id_test", inplace=True)

    extended.to_csv(
        "./data/02_intermediate/postulaciones/postulaciones_train_rank.csv",
        index=False,
    )


if __name__ == "__main__":
    # remove_duplicates()
    # clean()
    # join_train()
    # calculate_applicants_matrix()
    # generate_test()
    add_rank()
