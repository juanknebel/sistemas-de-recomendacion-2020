import pandas as pd
import surprise as sp
import logging


logger = logging.getLogger("surprise")
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


def _tune_train(
    X_train, y_train, prediction_algorithm, param_grid, scale, n_jobs=-1
):
    logger.info(f"Tunning {prediction_algorithm.__name__} ...")
    reader = sp.reader.Reader(rating_scale=scale)
    X_train_surprise = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    data_train = sp.dataset.Dataset.load_from_df(X_train_surprise, reader)

    gs = sp.model_selection.search.RandomizedSearchCV(
        prediction_algorithm,
        param_grid,
        measures=["rmse"],
        cv=5,
        n_jobs=n_jobs,
    )
    gs.fit(data_train)

    best_params = gs.best_params["rmse"]
    logger.debug(f"Best score in validation: {gs.best_score['rmse']}")
    logger.debug(f"Best params in validation: {best_params}")

    return best_params


def _train(train, prediction_algorithm, params, scale):
    logger.info(f"Training {prediction_algorithm.__name__} ...")
    reader = sp.reader.Reader(rating_scale=scale)
    data = sp.dataset.Dataset.load_from_df(train, reader)
    trainset, testset = sp.model_selection.train_test_split(
        data, test_size=0.30, random_state=0
    )

    model = prediction_algorithm(**params)
    model.fit(trainset)
    y_predictions = model.test(testset)
    logger.debug(f"Accuracy in test: {sp.accuracy.rmse(y_predictions)}")

    model.fit(data.build_full_trainset())
    return model


def _fit(test_pairs, model):
    return list(map(lambda x: model.predict(x[0], x[1]).est, test_pairs))


def _generate_submission(submission, file_name):
    submission.to_csv(file_name, index=False)
