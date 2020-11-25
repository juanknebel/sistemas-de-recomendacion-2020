import src.surprise_model
import surprise as sp
from src.utils import split_in
from src.predictors import PredictKNN, PredictSVD
from tqdm import tqdm
import pandas as pd
import numpy as np


def clean_books(df_books):
    df_books.loc[
        (df_books.idioma.isna()) & (df_books.isbn == "Español"), "idioma"
    ] = "Español"
    df_books = df_books[df_books.anio.notna()]
    df_books.loc[df_books.anio == "(200", "anio"] = "2002"
    not_years = list(
        filter(
            lambda y: len(y) != 4, [x for x in df_books.anio.unique().tolist()]
        )
    )
    df_books = df_books[~(df_books.anio.isin(not_years))]
    df_books.loc[df_books.isbn == "Español", "isbn"] = np.nan

    return df_books


def generate_new_features(df_train, algorithm, best_params):
    """
    Generación de nueva columna
    Separo en 5 el dataset, entreno con 4 partes y hago prediccion sobre el restante
    * [1,2,3,4] --> [5]
    * [1,2,3,5] --> [4]
    * [1,2,4,5] --> [3]
    * [1,3,4,5] --> [2]
    * [2,3,4,5] --> [1]
    """
    scale = (1.0, 10.0)
    index_split = split_in(list(np.arange(len(df_train))), 5)
    predictions = []

    for i in tqdm(range(5)):
        partial_df = pd.DataFrame()
        partial_test = df_train[df_train.index.isin(index_split[i])]
        for j in range(5):
            if j != i:
                partial_df = partial_df.append(
                    df_train[df_train.index.isin(index_split[j])]
                )
        model = src.surprise_model._train(
            partial_df[["usuario", "libro", "puntuacion"]],
            algorithm,
            best_params,
            scale,
        )
        predictions_test = src.surprise_model._fit(
            zip(partial_test.usuario, partial_test.libro), model
        )
        predictions += zip(index_split[i], predictions_test)

    return sorted(predictions, key=lambda x: x[0])


def generate_features_svd_knn_surprise():
    df_train = pd.read_csv("./data/01_raw/opiniones_train.csv")
    df_test = pd.read_csv("./data/01_raw/opiniones_test.csv")
    np.random.seed(0)

    svd_predictor = PredictSVD()
    svd_predictor.train(df_train)
    df_test["svd"] = svd_predictor.predict(df_test)

    knn_predictor = PredictKNN()
    knn_predictor.train(df_train)
    df_test["knn"] = knn_predictor.predict(df_test)

    svd = sp.prediction_algorithms.SVD
    knn = sp.prediction_algorithms.knns.KNNBasic
    features_svd = generate_new_features(
        df_train, svd, svd_predictor.best_params()
    )
    features_knn = generate_new_features(
        df_train, knn, knn_predictor.best_params()
    )
    df_train["svd"] = [pred[1] for pred in features_svd]
    df_train["knn"] = [pred[1] for pred in features_knn]

    df_train.to_csv(
        "./data/02_intermediate/opiniones_train_modelos.csv", index=False
    )
    df_test.to_csv(
        "./data/02_intermediate/opiniones_test_modelos.csv", index=False
    )


def add_mean(df, field):
    book_mean = df.groupby("libro")[field].mean().reset_index()
    book_mean.rename(columns={field: f"mean_{field}"}, inplace=True)
    df = df.merge(book_mean, how="left", left_on="libro", right_on="libro")
    return df


def add_genre_and_years(train, test, books):
    train = train.merge(
        books[["libro", "anio", "genero"]],
        how="left",
        left_on="libro",
        right_on="libro",
    )
    test = test.merge(
        books[["libro", "anio", "genero"]],
        how="left",
        left_on="libro",
        right_on="libro",
    )
    yaers_mode_train = int(train.anio.mode())
    yaers_mode_test = int(test.anio.mode())
    train.loc[train.anio.isna() == True, "anio"] = yaers_mode_train
    train.loc[train.genero.isna() == True, "genero"] = "NA"
    test.loc[test.anio.isna() == True, "anio"] = yaers_mode_test
    test.loc[test.genero.isna() == True, "genero"] = "NA"

    return train, test


def add_model_means(train, test):
    train = add_mean(train, "puntuacion")
    train = add_mean(train, "svd")
    train = add_mean(train, "knn")

    mean_puntuacion = train["mean_puntuacion"].mean()
    for name, _ in train.groupby(["libro", "mean_puntuacion"]):
        test.loc[test.libro == name[0], "mean_puntuacion"] = name[1]
    test.loc[test.mean_puntuacion.isna(), "mean_puntuacion"] = mean_puntuacion

    mean_svd = train["mean_svd"].mean()
    for name, _ in train.groupby(["libro", "mean_svd"]):
        test.loc[test.libro == name[0], "mean_svd"] = name[1]
    test.loc[test.mean_svd.isna(), "mean_svd"] = mean_svd

    mean_knn = train["mean_knn"].mean()
    for name, _ in train.groupby(["libro", "mean_knn"]):
        test.loc[test.libro == name[0], "mean_knn"] = name[1]
    test.loc[test.mean_knn.isna(), "mean_knn"] = mean_knn

    return train, test


def merge_features():
    df_users = pd.read_csv("./data/01_raw/usuarios.csv")
    df_books = pd.read_csv("./data/01_raw/libros.csv")
    generate_features_svd_knn_surprise()
    df_train = pd.read_csv("./data/02_intermediate/opiniones_train_modelos.csv")
    df_test = pd.read_csv("./data/02_intermediate/opiniones_test_modelos.csv")

    df_books = clean_books(df_books)
    df_train, df_test = add_genre_and_years(df_train, df_test, df_books)
    df_train, df_test = add_model_means(df_train, df_test)

    df_train.to_csv("./data/03_primary/opiniones_train_final.csv", index=False)
    df_test.to_csv("./data/03_primary/opiniones_test_final.csv", index=False)
    