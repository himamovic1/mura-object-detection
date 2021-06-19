import logging

from config.config import Config


def init_logging():
    log_formatter = logging.Formatter("%(asctime)s - [%(filename)s::%(lineno)d] -- %(levelname)s: %(message)s")

    app_config = Config()

    # create file handler which logs even debug messages
    file_handler = logging.FileHandler(app_config.LOGGING_FILE)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(app_config.LOGGING_LEVEL)

    # create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(app_config.LOGGING_LEVEL)

    logger = logging.getLogger()
    logger.setLevel(app_config.LOGGING_LEVEL)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging successfully initialised")
    return logger
