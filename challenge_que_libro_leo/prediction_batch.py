from prediction import start, ensamble
import logging
import config_with_yaml as config


logger = logging.getLogger("prediction-batch")
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
    cfg = config.load("config.yml")

    path_inut = cfg.getProperty("Challenge.LibroQueLeo.Path.input")
    train_file = path_inut + cfg.getProperty(
        "Challenge.LibroQueLeo.Files.train"
    )
    test_file = path_inut + cfg.getProperty("Challenge.LibroQueLeo.Files.test")

    output_path = cfg.getProperty("Challenge.LibroQueLeo.Path.submissions")
    submission_file = output_path + cfg.getProperty(
        "Challenge.LibroQueLeo.Files.submission"
    )

    model_path = cfg.getProperty("Challenge.LibroQueLeo.Path.models")
    model_file = model_path + cfg.getProperty(
        "Challenge.LibroQueLeo.Files.model"
    )

    experiment_description = cfg.getProperty("Challenge.LibroQueLeo.experiment")
    logger.info(f"Experiment description: {experiment_description}")

    method = cfg.getProperty("Challenge.LibroQueLeo.Predictor.method")

    random_state = cfg.getProperty("Challenge.LibroQueLeo.randomState")

    if method == "ensamble":
        # (predictor_identifier, weight), model_file
        weights = cfg.getProperty("Challenge.LibroQueLeo.Ensamble.Weights")
        models = cfg.getProperty("Challenge.LibroQueLeo.Ensamble.Models")
        model_tuples = []
        for predictor, weight in weights.items():
            model_file = model_path + models[predictor]
            model_tuples += [((predictor, weight), model_file)]

        ensamble(
            model_tuples,
            train_file,
            test_file,
            submission_file,
            logger,
            random_state,
        )
    else:
        start(
            method,
            train_file,
            test_file,
            submission_file,
            logger,
            model_file=model_file,
            random_state=random_state,
        )
