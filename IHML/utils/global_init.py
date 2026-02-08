##Avoid unwanted warnings/messages
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


##LOGGER related initialization
import logging
import sys
import datetime

#Log File is named with DAY_HOUR_MINUTE
print("Initializing logger")
logging.getLogger('numexpr').setLevel(logging.WARNING)
now = datetime.datetime.now()
filename = "experiment_"+str(now.year)+"_"+str(now.month)+"_"+str(now.day)

#Basic message format with message&time
log_format_std= '%(message)s'
log_format_file= '[%(asctime)s] %(message)s'
formatter_std=logging.Formatter(fmt=log_format_std)
formatter_file=logging.Formatter(fmt=log_format_file)

#STDOUT and FILE handlers
file_handler = logging.FileHandler(filename='experiments/'+filename)
file_handler.setFormatter(formatter_file)
file_handler.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter_std)
stdout_handler.setLevel(logging.INFO)

#Clear logger handler cash
logger = logging.getLogger()
if (logger.hasHandlers()):
    logger.handlers.clear()

#logger = logging.getLogger('INCREMENTAL_ML')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)