import torch
import os
import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.path.expanduser('~')
TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
SAVEPATH = '/home/adryw/dataset/ehdqn/ckpts/'
SAVEPATH = './ckpts/' if not os.path.isdir(SAVEPATH) else SAVEPATH
SAVEPATH = os.path.join(SAVEPATH, TIME)
LOGPATH = HOME + '/dataset/ehdqn/logs/' + TIME
#LOGPATH = '../logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
