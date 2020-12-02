import surprise as sp
from src.utils import split_in
from src import PredictKNN, PredictSVD
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle


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


def generate_new_features(df_train, predictor):
    """
    Generación de nueva columna
    Separo en 5 el dataset, entreno con 4 partes y hago prediccion sobre el restante
    * [1,2,3,4] --> [5]
    * [1,2,3,5] --> [4]
    * [1,2,4,5] --> [3]
    * [1,3,4,5] --> [2]
    * [2,3,4,5] --> [1]
    """
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

        X_train = partial_df[["usuario", "libro"]]
        y_train = partial_df.puntuacion
        predictor.tune(X_train, y_train)
        predictor.train(X_train, y_train)
        y_prediction = predictor.predict(
            predictor.transform_to_predict(partial_test[["usuario", "libro"]])
        )

        predictions += zip(index_split[i], y_prediction)

    return sorted(predictions, key=lambda x: x[0])


def load_model(input):
    return pickle.load(open(input, "rb"))


def generate_features_svd_knn_surprise():
    df_train = pd.read_csv("./data/01_raw/opiniones_train.csv")
    df_test = pd.read_csv("./data/01_raw/opiniones_test.csv")
    np.random.seed(0)

    svd_predictor = PredictSVD()
    svd_predictor.load_model(load_model("./data/06_models/svd"))
    df_test["svd"] = svd_predictor.predict(
        svd_predictor.transform_to_predict(df_test)
    )

    knn_predictor = PredictKNN()
    knn_predictor.load_model(load_model("./data/06_models/knn"))
    df_test["knn"] = knn_predictor.predict(
        knn_predictor.transform_to_predict(df_test)
    )

    df_train["svd"] = [
        feature[1] for feature in generate_new_features(df_train, PredictSVD())
    ]
    
    df_train["knn"] = [
        feature[1] for feature in generate_new_features(df_train, PredictKNN())
    ]

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


def add_opinions():
    df_train = pd.read_csv("./data/01_raw/opiniones_train.csv")
    df_test = pd.read_csv("./data/01_raw/opiniones_test.csv")

    opinions_by_user = (
        df_train.groupby("usuario")
        .agg("count")
        .reset_index()[["usuario", "libro"]]
        .rename(columns={"libro": "cantidad_opiniones"})
    )
    opinions_by_user_test = (
        df_test.groupby("usuario")
        .agg("count")
        .reset_index()[["usuario", "libro"]]
        .rename(columns={"libro": "cantidad_opiniones"})
    )

    opinions_by_user = (
        df_train.groupby("usuario")
        .agg("count")
        .reset_index()[["usuario", "libro"]]
        .rename(columns={"libro": "cantidad_opiniones"})
    )
    opinions_by_user_test = (
        df_test.groupby("usuario")
        .agg("count")
        .reset_index()[["usuario", "libro"]]
        .rename(columns={"libro": "cantidad_opiniones"})
    )

    # Cada dataset completo con la cantidad de opiniones
    new_train = df_train.merge(
        opinions_by_user, left_on="usuario", right_on="usuario"
    )
    new_test = df_test.merge(
        opinions_by_user_test, left_on="usuario", right_on="usuario", how="left"
    )

    new_train.to_csv(
        "./data/02_intermediate/opiniones_train_opiniones_1.csv", index=False
    )
    new_test.to_csv(
        "./data/02_intermediate/opiniones_test_opiniones_1.csv", index=False
    )

    # Pongo en test la cantidad de opiniones de train y completo los na con 0
    new_test_2 = df_test.merge(
        opinions_by_user, left_on="usuario", right_on="usuario", how="left"
    )

    new_test_2["cantidad_opiniones"] = new_test_2["cantidad_opiniones"].fillna(
        0
    )

    new_test_2.to_csv(
        "./data/02_intermediate/opiniones_test_opiniones_2.csv", index=False
    )


if __name__ == "__main__":
    # add_opinions()
    generate_features_svd_knn_surprise()
