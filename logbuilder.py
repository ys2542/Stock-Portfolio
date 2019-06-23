import os
import logging
from datetime import datetime



def buildLogger(log_dir, log_name, levelMain = logging.INFO):
    date = datetime.today().strftime('%Y%m%d')
    logger = logging.getLogger(log_name)
    logger.setLevel(level = levelMain)
    info_handler = logging.FileHandler(os.path.join(log_dir,date + "_info.log"))
    info_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    debug_handler = logging.FileHandler(os.path.join(log_dir, date + "_debug.log"))
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    error_handler = logging.FileHandler(os.path.join(log_dir,date + "_error.log"))
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    return logger