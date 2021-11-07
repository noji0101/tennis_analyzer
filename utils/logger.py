"""Logger"""
import sys
from logging import INFO, DEBUG, FileHandler
from logging import StreamHandler, Formatter, getLogger

def setup_logger(logfile: str = './logs/info.log'):
    """Set up Logger
    Parameters
    ----------
    logfile : str, optional
        logfile path, by default './logs/info.log'
    """
    logger = getLogger()
    logger.setLevel(INFO)

    # create file handler
    fh = FileHandler(logfile)
    fh.setLevel(INFO)
    fh_formatter = Formatter(fmt='')
    fh.setFormatter(fh_formatter)

    # create console handler
    ch = StreamHandler(stream=sys.stdout)
    ch.setLevel(INFO)
    ch_formatter = Formatter(fmt='')
    ch.setFormatter(ch_formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

def get_logger(name: str):
    """Logs a message
    Parameters
    ----------
    name : str
        name of logger
    """
    logger = getLogger(name)
    return logger