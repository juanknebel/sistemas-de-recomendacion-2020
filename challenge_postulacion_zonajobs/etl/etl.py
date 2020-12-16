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
    # Remove educacion
    df_applicants_education = pd.read_csv(
        "./data/01_raw/postulantes/postulantes_educacion.csv"
    )
    df_applicants_education.drop_duplicates(inplace=True)
    df_applicants_education.to_csv(
        "./data/02_intermediate/postulantes/postulantes_educacion.csv",
        index=False,
    )

    # Remove genero
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

    # Remove postulaciones
    df_applicants_train = pd.read_csv(
        "./data/01_raw/postulaciones/postulaciones_train.csv"
    )
    df_applicants_train.drop_duplicates(
        subset=["idaviso", "idpostulante"], inplace=True
    )
    df_applicants_train.to_csv(
        "./data/02_intermediate/postulaciones/postulaciones_train.csv",
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


@log(logger)
def join_notices():
    df_notice_detail = pd.read_csv("./data/01_raw/avisos/avisos_detalle.csv")

    df_notice_online = pd.read_csv("./data/01_raw/avisos/avisos_online.csv")

    df_notice_online = df_notice_online.merge(df_notice_detail, on="idaviso")
    df_notice_online.drop(
        columns=[
            "idpais",
            "ciudad",
            "mapacalle",
        ],
        inplace=True,
    )

    df_notice_online.to_csv(
        "./data/02_intermediate/avisos/avisos_detalle.csv", index=False
    )


@log(logger)
def add_age():
    df_notice_detail = pd.read_csv(
        "./data/02_intermediate/avisos/avisos_detalle.csv"
    )

    dtypes = {"idaviso": "int64", "idpostulante": "string"}
    mydateparser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df_applicants_train = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_train.csv",
        parse_dates=["fechapostulacion"],
        date_parser=mydateparser,
        dtype=dtypes,
    )

    df_applicants_genre_year = pd.read_csv(
        "./data/02_intermediate/postulantes/postulantes_genero_edad.csv"
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

    df_applicants_train = df[["idaviso", "idpostulante", "fechapostulacion"]]
    df_applicants_train.to_csv(
        "./data/02_intermediate/postulaciones/postulaciones_train.csv",
        index=False,
    )


@log(logger)
def generate_test():
    df_applicants_to_predict = pd.read_csv(
        "./data/01_raw/ejemplo_de_solucion.csv"
    ).drop_duplicates(subset=["idpostulante"])

    df_applicants_to_predict.to_csv(
        "./data/02_intermediate/postulaciones/postulaciones_test.csv",
        index=False,
    )


def matrix_applicants_metada(
    applicants: pd.DataFrame,
    applicants_genre_year: pd.DataFrame,
    applicants_education: pd.DataFrame,
):
    df = applicants.merge(applicants_genre_year, on="idpostulante").merge(
        applicants_education, on="idpostulante", how="left"
    )
    df["value"] = 1
    df.drop(columns="fechanacimiento", inplace=True)

    matrix_metadata = df.pivot(
        index="idpostulante",
        columns=["sexo", "nombre", "estado"],
        values="value",
    ).fillna(0)
    matrix_metadata.columns = matrix_metadata.columns.to_flat_index()

    return matrix_metadata


@log(logger)
def calculate_applicants_matrix():
    df_applicants_train = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_train.csv"
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

    ranking_by_applicant = (
        df_applicants_train.groupby("idpostulante")
        .agg({"idaviso": "count"})
        .reset_index()
        .rename(columns={"idaviso": "cantidad"})
    )

    # Genero la matriz para saber similaridad con respecto a los hard users,
    # o sea, aquellos que tienen mas de 100 postulaciones
    matrix_train = matrix_applicants_metada(
        ranking_by_applicant[ranking_by_applicant.cantidad > 100],
        df_applicants_genre_year,
        df_applicants_education,
    )
    matrix_test = matrix_applicants_metada(
        df_applicants_to_predict,
        df_applicants_genre_year,
        df_applicants_education,
    )

    columns_names = set(matrix_test.columns.values) | set(matrix_train.columns)

    # Completo las columnas que le pueda llegar a faltar a cada matriz
    for a_column in columns_names:
        if a_column not in matrix_train.columns:
            matrix_train[a_column] = 0
        if a_column not in matrix_test.columns:
            matrix_test[a_column] = 0

    columns_dict = {}
    for index, test_tuple in enumerate(columns_names):
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
        "./data/02_intermediate/postulantes/atributos_diccionario.csv",
        index=False,
    )


@log(logger)
def add_rank():
    dtypes = {"idaviso": "int64", "idpostulante": "string"}
    mydateparser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df_applicants = pd.read_csv(
        "./data/02_intermediate/postulaciones/postulaciones_train.csv",
        parse_dates=["fechapostulacion"],
        date_parser=mydateparser,
        dtype=dtypes,
    )

    applicants_with_rank = []

    for _, group in tqdm(df_applicants.groupby("idpostulante")):
        temp = group.sort_values(by="fechapostulacion", ascending=False)
        temp["rank"] = range(len(temp))
        applicants_with_rank += [temp]

    df_applicants_with_rank = pd.concat(applicants_with_rank)

    df_applicants_with_rank.to_csv(
        "./data/02_intermediate/postulaciones/postulaciones_train_rank.csv",
        index=False,
    )


if __name__ == "__main__":
    remove_duplicates()
    clean()
    join_notices()
    add_age()
    generate_test()
    calculate_applicants_matrix()
    add_rank()
