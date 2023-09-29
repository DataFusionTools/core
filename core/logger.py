import logging
import sys


def get_generic_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler("datafusiontools.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


global LOGGER
LOGGER = get_generic_logger()


class Logger:
    def __getattribute__(self, item):
        try:
            value = object.__getattribute__(self, item)
            name = getattr(callable, "__name__", False)
            if callable(value) and name:
                LOGGER.info(f"Function called {name}")
        except:
            print(1)
