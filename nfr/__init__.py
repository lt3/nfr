import logging


def set_logger():
    # Do not needlessly expose variables
    logger = logging.getLogger("nfr")
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt="%(asctime)s - [%(levelname)s]: %(message)s", datefmt="%d-%b %H:%M:%S"))
    logger.addHandler(sh)


set_logger()

__version__ = "1.0.1"
