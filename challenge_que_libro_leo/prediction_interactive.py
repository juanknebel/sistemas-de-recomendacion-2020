from prediction import start
import argparse
import logging


logger = logging.getLogger("prediction-interactive")
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


if __name__ == "__main__":
    methods = ["predict_svd", "predict_knn", "predict_lightgbm", "ensamble"]

    execution_help = ", ".join(methods)
    parser = argparse.ArgumentParser(description="Prediction cli.")
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        type=str,
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

    method = args.method

    train_file = "./data/02_intermediate/opiniones_train_opiniones_1.csv"
    test_file = "./data/02_intermediate/opiniones_test_opiniones_2.csv"
    submission_file = f"./data/07_model_output/{args.output}.csv"
    logger.info(f"Experiment description: {args.experiment}")
    start(method, train_file, test_file, submission_file, logger)
