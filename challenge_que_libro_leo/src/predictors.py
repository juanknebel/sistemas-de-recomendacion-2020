from sklearn.metrics import mean_squared_error
import src.lightgbm_model
import src.surprise_model
import surprise as sp
import lightgbm
import math


class PredictionModel:
    def __init__(self):
        self.algorithm = None
        self.param_grid = {}
        self.n_jobs = 0
        self.best_params_ = None

    def predict(self, test):
        raise NotImplementedError

    def tune_and_train(self, train):
        raise NotImplementedError

    def train(self, train):
        raise NotImplementedError

    def best_params(self):
        return self.best_params_

    def rmse(self, y_test, y_predicted):
        return math.sqrt(mean_squared_error(y_test, y_predicted))


class PredictSurprise(PredictionModel):
    def __init__(self):
        super(PredictSurprise, self).__init__()

    def predict(self, test):
        return src.surprise_model._fit(
            zip(test.usuario, test.libro), self.model
        )

    def tune_and_train(self, train):
        scale = (1.0, 10.0)

        self.best_params_ = src.surprise_model._tune_train(
            train[["usuario", "libro"]],
            train[["puntuacion"]],
            self.algorithm,
            self.param_grid,
            scale,
            n_jobs=self.n_jobs,
        )

        self.train(train)

    def train(self, train):
        scale = (1.0, 10.0)

        self.model = src.surprise_model._train(
            train[["usuario", "libro", "puntuacion"]],
            self.algorithm,
            self.best_params_,
            scale,
        )


class PredictSVD(PredictSurprise):
    def __init__(self):
        super(PredictSVD, self).__init__()
        self.algorithm = sp.prediction_algorithms.SVD
        self.n_jobs = -1
        self.param_grid = {
            "n_epochs": [300, 700, 1000],
            "lr_all": [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007],
            "reg_all": [0.2, 0.4, 0.42, 0.45, 0.49, 0.6],
            "n_factors": range(40, 130, 5),
        }


class PredictKNN(PredictSurprise):
    def __init__(self):
        super(PredictKNN, self).__init__()
        self.algorithm = sp.prediction_algorithms.knns.KNNBasic
        self.n_jobs = 6
        self.param_grid = {
            "k": [15, 40, 70],
            "min_k": [1, 3, 9],
            "sim_options": {"name": ["msd", "cosine"], "user_based": [False]},
        }


class PredictLightGbm(PredictionModel):
    def __init__(self):
        super(PredictLightGbm, self).__init__()
        self.algorithm = lightgbm.LGBMRegressor()
        self.n_jobs = -1
        self.param_grid = {
            "learning_rate": [0.001, 0.003, 0.01, 0.05],
            "num_leaves": [4, 15, 20],
            "boosting_type": ["gbdt", "goss", "dart"],
            "max_depth": [8, 15, 20, 31],
            "random_state": [42],
            "n_estimators": [100, 500, 1000],
            "colsample_bytree": [0.1, 0.5, 0.9],
            "subsample": [0.7, 0.9],
            "max_bin": [1, 5, 14, 20],
            #    'min_split_gain' : [0.01],
            #    'min_data_in_leaf':[5, 10, 15],
            "metric": ["rmse"],
            "reg_alpha": [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
            "reg_lambda": [0, 1e-1, 1, 5, 10, 20, 50, 100],
            #'early_stopping_round': [100, 500],
            "min_data_in_leaf": [10, 20, 40],
            "min_sum_hessian_in_leaf": [0.0001, 0.0004, 0.001, 0.1],
            "n_iter": [10, 30, 100],
        }

    def predict(self, test):
        return src.lightgbm_model._fit(test, self.algorithm)

    def tune_and_train(self, train):
        self.best_params_ = src.lightgbm_model._tune_train(
            train.drop(columns="puntuacion", axis=1),
            train.puntuacion,
            self.algorithm,
            self.param_grid,
            n_jobs=self.n_jobs,
        )
        self.train(train)

    def train(self, train):
        self.algorithm.set_params(**self.best_params_)
        src.lightgbm_model._train(
            train.drop(columns="puntuacion", axis=1),
            train.puntuacion,
            self.algorithm,
        )
