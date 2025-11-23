import logging
from logging.config import dictConfig

DEFAULT_LOGGING_CONFIG = {
    "handlers": {
        "muse": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "muse": {
            "handlers": ["muse"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False
}

def init_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)
    return logger

def _configure_logging(config: dict = DEFAULT_LOGGING_CONFIG):
    dictConfig(config)

_configure_logging(DEFAULT_LOGGING_CONFIG)