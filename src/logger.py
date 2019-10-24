from tensorboard_logging import Logger
import datetime
import os

HOME = os.path.expanduser('~')
LOGPATH = HOME + '/dataset/ehdqn/logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#LOGPATH = '../logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TB_LOGGER = Logger(LOGPATH)
