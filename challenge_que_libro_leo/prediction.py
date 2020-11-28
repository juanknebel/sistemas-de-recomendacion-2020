from src import PredictSVD, PredictKNN, PredictLightgbm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import logging
import math
import pickle


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


def save_model(model, output):
    pickle.dump(model, open(output, "wb"))


def predict(predictor, X_train, X_test, y_train, y_test, to_predict):
    predictor.tune(X_train, y_train)
    predictor.train(X_train, y_train)
    y_prediction = predictor.predict(predictor.transform_to_predict(X_test))
    rmse = predictor.rmse(y_test, y_prediction)
    logger.info(f"{predictor.__class__.__name__} in test: {rmse}")

    all_x = pd.concat([X_train, X_test], axis=0)
    all_y = pd.concat([y_train, y_test], axis=0)
    predictor.train(all_x, all_y)

    return predictor.predict(predictor.transform_to_predict(to_predict))


def predict_surprise(predictor, X_train, X_test, y_train, y_test, to_predict):
    return predict(
        predictor,
        X_train[["usuario", "libro"]],
        X_test[["usuario", "libro"]],
        y_train,
        y_test,
        to_predict,
    )


def predict_svd(X_train, X_test, y_train, y_test, to_predict):
    return predict_surprise(
        PredictSVD(), X_train, X_test, y_train, y_test, to_predict
    )


def predict_knn(X_train, X_test, y_train, y_test, to_predict):
    return predict_surprise(
        PredictKNN(), X_train, X_test, y_train, y_test, to_predict
    )


def predict_lightgbm(X_train, X_test, y_train, y_test, to_predict):
    return predict(
        PredictLightgbm(),
        X_train,
        X_test,
        y_train,
        y_test,
        to_predict.drop(columns=["id", "puntuacion"], axis=1),
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
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        nargs=None,
        help=f"name of the file that will have the final submission",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--experiment",
        dest="experiment",
        type=str,
        nargs=None,
        help="description of the experiment",
        required=True,
    )
    args = parser.parse_args()

    method_number = args.method
    if method_number != 1 and args.output == None:
        raise Exception("You must specified filename for the subission")

    else:
        method = pipeline_executions[method_number][0]
        df_train = pd.read_csv(
            "./data/02_intermediate/opiniones_train_opiniones_1.csv"
        )
        df_test = pd.read_csv(
            "./data/02_intermediate/opiniones_test_opiniones_2.csv"
        )
        df_train["libro"] = df_train["libro"].astype("category")
        df_test["libro"] = df_test["libro"].astype("category")
        # df_train["genero"] = df_train["genero"].astype("category")
        # df_test["genero"] = df_test["genero"].astype("category")
        # df_train["anio"] = df_train["anio"].astype("int")
        # df_test["anio"] = df_test["anio"].astype("int")

        logger.info(f"Experiment description: {args.experiment}")
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
                        np.round(prediction, 4),
                    )
                ),
            }
        )

        submission.to_csv(
            f"./data/07_model_output/{args.output}.csv", index=False
        )
