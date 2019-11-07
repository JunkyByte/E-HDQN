import torch
import os
import datetime
import numpy as np
torch.manual_seed(42)
np.random.seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.path.expanduser('~')
TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
SAVEPATH = '/home/adryw/dataset/ehdqn/ckpts/'
SAVEPATH = '../ckpts/' if not os.path.isdir(SAVEPATH) else SAVEPATH
SAVEPATH = os.path.join(SAVEPATH, TIME)
LOGPATH = HOME + '/dataset/ehdqn/logs/' + TIME
#LOGPATH = '../logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
