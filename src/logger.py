from tensorboard_logging import Logger
import datetime


LOGPATH = '~/dataset/ehdqn/logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#LOGPATH = '../logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TB_LOGGER = Logger(LOGPATH)