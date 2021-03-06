from surprise import SVDpp, SVD, KNNBasic, KNNWithMeans, NMF
from surprise import Dataset
from surprise.model_selection.search import RandomizedSearchCV
from surprise.reader import Reader
from surprise.model_selection import cross_validate, GridSearchCV
import pandas as pd
import logging
import numpy as np
import pickle
import sys


logger = logging.getLogger("prediction-batch")
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


def testing_algorithm(train):
    for algo in [SVD(), SVDpp(), KNNWithMeans(), KNNBasic()]:
        # Run 5-fold cross-validation and print results.
        cross_validate(
            algo,
            train,
            measures=["RMSE", "MAE"],
            cv=5,
            verbose=True,
            n_jobs=-1,
        )


def predict(algo, data, col_name: str = "puntuacion"):
    usr_book_tuple = zip(
        data[["id", "usuario", "genero", "libro"]].iloc[:, 0],
        data[["id", "usuario", "genero", "libro"]].iloc[:, 1],
        data[["id", "usuario", "genero", "libro"]].iloc[:, 2],
        data[["id", "usuario", "genero", "libro"]].iloc[:, 3],
    )
    predictions = []
    for id, user, genre, book in usr_book_tuple:
        rating_est = np.round(algo.predict(user, book).est, 4)
        rating_est = min(max(rating_est, 1), 10)
        predictions += [(id, user, genre, book, rating_est)]

    submission = pd.DataFrame(
        predictions, columns=["id", "usuario", "genero", "libro", col_name]
    )

    return submission


def tune(
    algo, data_train, param_grid: dict = {"random_state": [0]}, n_jobs: int = -1
):
    gs = GridSearchCV(
        algo, param_grid, measures=["rmse", "mae"], cv=5, n_jobs=n_jobs
    )
    gs.fit(data_train)
    estimator = gs.best_estimator["rmse"]
    best_score = gs.best_score["rmse"]
    best_params = gs.best_params["rmse"]
    logger.info(f"{algo.__name__} best score: {best_score}")
    logger.info(f"{algo.__name__} best params: {best_params}")

    # eval = cross_validate(estimator, data_train, measures=["RMSE", "MAE"])
    return estimator


def replace_usr_unknows_with_mean(
    # ["id", "usuario", "genero", "libro", "<algo>_puntuacion"]
    predicted: pd.DataFrame, 
    users_in_train: set,
    genre_means: dict,
    genre_book_means: dict,
):

    ids_not_in_train = predicted[~predicted.usuario.isin(users_in_train)].id.values
    for the_id in ids_not_in_train:
        row = predicted[predicted.id == the_id]
        try:
            the_mean = genre_book_means[(row.genero.values[0], row.libro.values[0])]
        except:
            the_mean = genre_means[row.genero.values[0]]
        predicted.iloc[~predicted.id == the_id, 4] = the_mean

    return predicted


def save_model(model, output: str):
    pickle.dump(model, open(output, "wb"))


def calculate_means(train: pd.DataFrame):
    mean_genre = train.groupby("genero").agg({"puntuacion": "mean"})
    mg = mean_genre.to_dict()["puntuacion"]
    mean_genre_book = train.groupby(["genero", "libro"]).agg(
        {"puntuacion": "mean"}
    )
    # mean_genre_book.loc[('Hombre', '10-anos-con-mafalda'), :]
    mgb = mean_genre_book.to_dict()["puntuacion"]

    return mg, mgb


if __name__ == "__main__":
    identificator = sys.argv[1]
    logger.info(f"Start experiment number: {identificator}")

    train_directory = "./data/02_intermediate/"
    submission_directory = "./data/07_model_output/"
    model_directory = "./data/06_models/"
    file_train = f"{train_directory}opiniones_train.csv"
    file_test = f"{train_directory}opiniones_test.csv"

    train = pd.read_csv(file_train)
    test = pd.read_csv(file_test)

    genre_means, genre_book_means = calculate_means(train)
    global_mean = train.puntuacion.mean()
    users_in_train = set(train.usuario.values)

    scale = (1.0, 10.0)
    reader = Reader(rating_scale=scale)
    data_train = Dataset.load_from_df(
        train[["usuario", "libro", "puntuacion"]], reader
    )
    trainset = data_train.build_full_trainset()

    # testing_algorithm(train)

    # SVD
    param_grid = {
        "n_factors": [70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
        "n_epochs": [100],
        "lr_all": [0.002, 0.005, 0.01, 0.05],
        "reg_all": [0.1, 0.4, 0.6],
        "random_state": [0, 5, 42],
    }

    svd = tune(SVD, data_train, param_grid, -1)
    svd.fit(trainset)
    svd_predict = predict(svd, test, "svd_puntuacion")

    svd_predict = replace_usr_unknows_with_mean(
        svd_predict, users_in_train, genre_means, genre_book_means
    )
    save_model(svd, f"{model_directory}svd_{identificator}")
    svd_predict[["id", "svd_puntuacion"]].to_csv(
        f"{submission_directory}svd_{identificator}.csv", index=False
    )

    # KNN
    param_grid = {
        "k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "sim_options": {"user_based": [True, False]},
        "bsl_options": {"method": ["als", "sgd"]},
        "random_state": [0, 5, 42],
    }

    knn = tune(KNNBasic, data_train, param_grid, 8)
    knn.fit(trainset)
    knn_predict = predict(knn, test, "knn_puntuacion")

    knn_predict = replace_usr_unknows_with_mean(
        knn_predict, users_in_train, genre_means, genre_book_means
    )
    save_model(knn, f"{model_directory}knn_{identificator}")
    knn_predict[["id", "knn_puntuacion"]].to_csv(
        f"{submission_directory}knn_{identificator}.csv", index=False
    )

    # NMFb
    param_grid = {
        "n_factors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "n_epochs": [100],
        "biased": [True],
        "reg_bu": [0.0001, 0.1, 0.3, 0.6],
        "reg_bi": [0.0001, 0.1, 0.3, 0.6],
        "random_state": [0, 5, 42],
    }

    nmfb = tune(NMF, data_train, param_grid, -1)
    nmfb.fit(trainset)
    nmfb_predict = predict(nmfb, test, "nmfb_puntuacion")

    nmfb_predict = replace_usr_unknows_with_mean(
        nmfb_predict, users_in_train, genre_means, genre_book_means
    )
    save_model(nmfb, f"{model_directory}nmfb_{identificator}")
    nmfb_predict[["id", "nmfb_puntuacion"]].to_csv(
        f"{submission_directory}nmfb_{identificator}.csv", index=False
    )

    final_predict = (
        svd_predict[["id", "svd_puntuacion"]]
        .merge(knn_predict[["id", "knn_puntuacion"]], on="id", how="inner")
        .merge(nmfb_predict[["id", "nmfb_puntuacion"]], on="id", how="inner")
    )

    final_predict["puntuacion"] = final_predict[
        ["svd_puntuacion", "knn_puntuacion", "nmfb_puntuacion"]
    ].mean(axis=1)

    final_predict.to_csv(
        f"{submission_directory}all_{identificator}.csv", index=False
    )
