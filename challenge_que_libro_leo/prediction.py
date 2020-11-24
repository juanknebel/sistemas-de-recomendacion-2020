from src.predictors import PredictSVD, PredictKNN, PredictLightGbm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import logging
import math


logger = logging.getLogger("prediction")
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


def predict_svd(X_train, X_test, y_train, y_test, to_predict):
    predictor = PredictSVD()
    train = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    predictor.tune_and_train(train)
    y_prediction = predictor.predict(X_test)
    rmse = predictor.rmse(y_test, y_prediction)
    logger.info(f"svd_rmse in test: {rmse}")

    test = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
    predictor.train(pd.concat([train, test]))

    return predictor.predict(to_predict)


def predict_knn(X_train, X_test, y_train, y_test, to_predict):
    predictor = PredictKNN()
    train = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    predictor.tune_and_train(train)
    y_prediction = predictor.predict(X_test)
    rmse = predictor.rmse(y_test, y_prediction)
    logger.info(f"knn_rmse in test: {rmse}")

    test = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
    predictor.train(pd.concat([train, test]))

    return predictor.predict(to_predict)


def predict_lightgbm(X_train, X_test, y_train, y_test, to_predict):
    predictor = PredictLightGbm()
    train = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    predictor.tune_and_train(train)
    y_prediction = predictor.predict(X_test)
    rmse = predictor.rmse(y_test, y_prediction)
    logger.info(f"lightgbm_rmse in test: {rmse}")

    test = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
    predictor.train(pd.concat([train, test]))

    return predictor.predict(
        to_predict.drop(columns=["id", "puntuacion"], axis=1)
    )


if __name__ == "__main__":
    pipeline_executions = {
        1: [predict_svd],
        2: [predict_knn],
        3: [predict_lightgbm],
    }

    execution_help = ""
    for k, v in pipeline_executions.items():
        execution_help = (
            execution_help
            + f"Method {k}: {', '.join([m.__name__ for m in v])}.\n"
        )
    parser = argparse.ArgumentParser(description="Prediction cli.")
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        type=int,
        nargs=None,
        help=f"which prediction method execute. {execution_help}",
    )
    parser.add_argument(
        "-p",
        "--prediction",
        dest="prediction",
        type=str,
        nargs=None,
        help=f"Name of the file that will have the final submission, not necessary to generate_features_svd_knn_surprise nor merge_features",
        required=False,
    )
    args = parser.parse_args()

    method_number = args.method
    if method_number != 1 and args.prediction == None:
        raise Exception("You must specified filename for the subission")

    else:
        method = pipeline_executions[method_number][0]
        if method_number == 1 or method_number == 2:
            method()
        else:
            df_train = pd.read_csv("./data/01_raw/opiniones_train.csv")
            df_test = pd.read_csv("./data/01_raw/opiniones_test.csv")
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

            prediction = method(X_train, X_test, y_train, y_test, df_test)

            submission = pd.DataFrame(
                {
                    "id": df_test.id,
                    "puntuacion": list(
                        map(
                            lambda p: 10.0
                            if p >= 10.0
                            else (1.0 if p <= 0.0 else p),
                            np.round(prediction, 0),
                        )
                    ),
                }
            )

            submission.to_csv(
                f"./data/07_model_output/{args.prediction}.csv", index=False
            )
