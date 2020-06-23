import logging
from logging import handlers as loghand

def log_setup():

    logger = logging.getLogger()
    debug_handler = loghand.TimedRotatingFileHandler(filename=r'C:\Users\kylea\OneDrive\Documents\Stock_DB\debug_log.log',\
                when='midnight', interval=1, backupCount=7)

    formatter = logging.Formatter('Process:%(process)s - %(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s')
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    logger.setLevel(logging.DEBUG)
    debug_handler.setLevel(logging.DEBUG)

def truncate_logger():
    with open(r'C:\Users\kylea\OneDrive\Documents\Stock_DB\debug_log.log', 'w'):
        pass
