import torch
import datetime
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVEPATH = '/home/adryw/dataset/ehdqn/ckpts/'
SAVEPATH = '../ckpts/' if not os.path.isdir(SAVEPATH) else SAVEPATH

print('Torch Device: %s' % device)
