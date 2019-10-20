import torch
import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVEPATH = '/home/adryw/dataset/ehdqn/ckpts/'
