from sklearn.metrics import mean_squared_error
import math


class PredictionModel:
    def __init__(self, random_state=0):
        self.param_grid = {}
        self.n_jobs = 0
        self.best_params_ = None
        self.random_state = random_state
        self.model_ = None

    def predict(self, to_predict):
        raise NotImplementedError

    def tune(self, X_train, y_train):
        raise NotImplementedError

    def train(self, X_train, y_train):
        raise NotImplementedError

    def best_params(self):
        return self.best_params_

    def rmse(self, y_real, y_predicted):
        return math.sqrt(mean_squared_error(y_real, y_predicted))

    def transform_to_predict(self, to_predict):
        return to_predict
