from src.predictors import PredictionModel
from src.utils import log
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import xgboost
import logging


logger = logging.getLogger("lightgbm")
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


class PredictXgboost(PredictionModel):
    def __init__(self, random_state=0):
        super(PredictXgboost, self).__init__(random_state)
        self.model_ = xgboost.XGBRegressor()
        self.n_jobs = -1
        self.param_grid = {
            "learning_rate": [0.001, 0.01, 0.1],
            "max_depth": [3, 5, 7, 10, 15, 24],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.5, 0.7, 0.9],
            "colsample_bytree": [0.5, 0.7],
            "n_estimators": [100, 200, 500, 2000],
            "objective": ["reg:squarederror"],
            "eval_metric": ["rmse"],
        }

    @log(logger)
    def predict(self, to_predict):
        return self.model_.predict(to_predict)

    @log(logger)
    def tune(self, train_x, train_y):
        X_train, X_validation, y_train, y_validation = train_test_split(
            train_x,
            train_y,
            test_size=0.3,
            random_state=self.random_state,
        )
        gs = RandomizedSearchCV(
            self.model_, self.param_grid, verbose=1, cv=5, n_jobs=self.n_jobs
        )

        gs.fit(X_train, y_train)
        logger.info(f"Best score tunning: {gs.best_score_}")
        logger.info(f"Best params tunning: {gs.best_params_}")
        self.best_params_ = gs.best_params_

        predicted = gs.predict(X_validation)
        rmse = self.rmse(y_validation, predicted)
        logger.info(f"rmse in validation: {rmse}")

    @log(logger)
    def train(self, X_train, y_train):
        self.model_.set_params(**self.best_params_)
        self.model_.fit(X_train, y_train)
