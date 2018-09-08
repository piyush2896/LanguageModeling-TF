from .config import config
import logging
import pickle

def get_tensorflow_logger(log_filename):
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.DEBUG)

    fhandler = logging.FileHandler(log_filename)
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(fhandler)
    return logger

def pickle_obj(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def unpickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
