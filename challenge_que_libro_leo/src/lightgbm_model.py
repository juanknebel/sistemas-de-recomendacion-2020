import numpy as np
import pandas as pd
import math
import logging

import lightgbm
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error



logger = logging.getLogger("lightgbm")
logger.setLevel(level=logging.DEBUG)

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


def _tune_train(X_train, y_train, prediction_algorithm, param_grid, n_jobs=-1):
    logger.info(f"Tunning lgbm ...")
    gs = RandomizedSearchCV(
        prediction_algorithm, param_grid, verbose=1, cv=5, n_jobs=n_jobs
    )
    gs.fit(X_train, y_train)
    best_params = gs.best_params_
    logger.debug(f"Best score in validation: {gs.best_score_}")
    logger.debug(f"Best params in validation: {best_params}")

    return best_params


def _train(train, train_y, prediction_algorithm):
    logger.info(f"Training lgbm ...")

    X_train, X_test, y_train, y_test = train_test_split(
        train, train_y, test_size=0.3, random_state=0
    )
    model = prediction_algorithm.fit(X_train, y_train)
    y_predictions = model.predict(X_test)

    logger.debug(
        f"Accuracy in test: {math.sqrt(mean_squared_error(y_test,y_predictions))}"
    )
    return model


def _fit(test, model):
    logger.info(f"Fitting test ...")
    return model.predict(test)


def _generate_submission(submission, file_name):
    logger.info(f"Saving submission to {file_name} ...")
    submission.to_csv(file_name, index=False)
