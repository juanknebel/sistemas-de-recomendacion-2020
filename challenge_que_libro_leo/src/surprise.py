from src.predictors import PredictionModel
from src.utils import log
import pandas as pd
import surprise as sp
import logging


logger = logging.getLogger("surprise")
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


class PredictSurprise(PredictionModel):
    def __init__(self, random_state=0):
        super(PredictSurprise, self).__init__(random_state)
        self.algorithm = sp.prediction_algorithms.algo_base.AlgoBase

    @log(logger)
    def predict(self, to_predict):
        return list(
            map(lambda x: self.model_.predict(x[0], x[1]).est, to_predict)
        )

    @log(logger)
    def tune(self, X_train, y_train):
        scale = (1.0, 10.0)
        reader = sp.reader.Reader(rating_scale=scale)
        X_train_surprise = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
        data_train = sp.dataset.Dataset.load_from_df(X_train_surprise, reader)
        gs = sp.model_selection.search.RandomizedSearchCV(
            self.algorithm,
            self.param_grid,
            measures=["rmse"],
            cv=5,
            n_jobs=self.n_jobs,
        )
        gs.fit(data_train)
        self.best_params_ = gs.best_params["rmse"]
        logger.info(f"Best score tunning: {gs.best_score['rmse']}")
        logger.info(f"Best params tunning: {self.best_params_}")

    @log(logger)
    def train(self, X_train, y_train):
        scale = (1.0, 10.0)
        reader = sp.reader.Reader(rating_scale=scale)
        X_train_surprise = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
        trainset = sp.dataset.Dataset.load_from_df(
            X_train_surprise, reader
        ).build_full_trainset()

        self.model_ = self.algorithm(**self.best_params_)
        self.model_.fit(trainset)

    def transform_to_predict(self, to_predict):
        return zip(to_predict.iloc[:,0], to_predict.iloc[:,1])


class PredictSVD(PredictSurprise):
    def __init__(self, random_state=0):
        super(PredictSVD, self).__init__(random_state)
        self.algorithm = sp.prediction_algorithms.SVD
        self.n_jobs = -1
        self.param_grid = {
            "n_epochs": [300, 700, 1000],
            "lr_all": [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007],
            "reg_all": [0.2, 0.4, 0.42, 0.45, 0.49, 0.6],
            "n_factors": range(40, 130, 5),
        }



class PredictKNN(PredictSurprise):
    def __init__(self, random_state=0):
        super(PredictKNN, self).__init__(random_state)
        self.algorithm = sp.prediction_algorithms.knns.KNNBasic
        self.n_jobs = 6
        self.param_grid = {
            "k": [15, 40, 70],
            "min_k": [1, 3, 9],
            "sim_options": {"name": ["msd", "cosine"], "user_based": [False]},
        }