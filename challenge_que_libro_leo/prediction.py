from src import PredictSVD, PredictKNN, PredictLightgbm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import pickle


def save_model(model, output):
    pickle.dump(model, open(output, "wb"))


def predict(predictor, X_train, X_test, y_train, y_test, to_predict, logger):
    predictor.tune(X_train, y_train)
    predictor.train(X_train, y_train)
    y_prediction = predictor.predict(predictor.transform_to_predict(X_test))
    rmse = predictor.rmse(y_test, y_prediction)
    logger.info(f"{predictor.__class__.__name__} in test: {rmse}")

    all_x = pd.concat([X_train, X_test], axis=0)
    all_y = pd.concat([y_train, y_test], axis=0)
    predictor.train(all_x, all_y)

    return predictor.predict(predictor.transform_to_predict(to_predict))


def predict_surprise(
    predictor, X_train, X_test, y_train, y_test, to_predict, logger
):
    return predict(
        predictor,
        X_train[["usuario", "libro"]],
        X_test[["usuario", "libro"]],
        y_train,
        y_test,
        to_predict,
        logger,
    )


def predict_svd(X_train, X_test, y_train, y_test, to_predict, logger):
    return predict_surprise(
        PredictSVD(), X_train, X_test, y_train, y_test, to_predict, logger
    )


def predict_knn(X_train, X_test, y_train, y_test, to_predict, logger):
    return predict_surprise(
        PredictKNN(), X_train, X_test, y_train, y_test, to_predict, logger
    )


def predict_lightgbm(X_train, X_test, y_train, y_test, to_predict, logger):
    return predict(
        PredictLightgbm(),
        X_train,
        X_test,
        y_train,
        y_test,
        to_predict.drop(columns=["id", "puntuacion"], axis=1),
        logger,
    )


def start(the_method, train_file, test_file, submission_file, logger):
    methods = {
        "predict_svd": predict_svd,
        "predict_knn": predict_knn,
        "predict_lightgbm": predict_lightgbm,
    }

    try:
        method = methods[the_method]
    except KeyError:
        raise ValueError("Invalid method")

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    df_train["libro"] = df_train["libro"].astype("category")
    df_test["libro"] = df_test["libro"].astype("category")
    # df_train["genero"] = df_train["genero"].astype("category")
    # df_test["genero"] = df_test["genero"].astype("category")
    # df_train["anio"] = df_train["anio"].astype("int")
    # df_test["anio"] = df_test["anio"].astype("int")

    X_train, X_test, y_train, y_test = train_test_split(
        df_train.drop(columns=["puntuacion"], axis=1),
        df_train.puntuacion,
        test_size=0.2,
        random_state=0,
    )

    prediction = method(X_train, X_test, y_train, y_test, df_test, logger)

    submission = pd.DataFrame(
        {
            "id": df_test.id,
            "puntuacion": list(
                map(
                    lambda p: 10.0 if p >= 10.0 else (1.0 if p <= 0.0 else p),
                    np.round(prediction, 4),
                )
            ),
        }
    )

    submission.to_csv(f"{submission_file}.csv", index=False)
