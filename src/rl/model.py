import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import settings as sett


class DDQN_Model(nn.Module):
    def __init__(self, state_size, action_size, conv, hidd_ch=256, conv_ch=32):
        super(DDQN_Model, self).__init__()
        self.action_size = action_size
        self.hidd_ch = hidd_ch
        state_size = np.array(state_size)[[3, 1, 2, 0]]

        if conv:
            self.features = nn.Sequential(
                nn.Conv2d(state_size[0], conv_ch, kernel_size=3, stride=3, padding=1),
                nn.ELU(),
                nn.Conv2d(conv_ch, conv_ch, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(conv_ch, conv_ch, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(conv_ch, conv_ch, kernel_size=3, stride=2, padding=1),
                nn.ELU()
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(state_size, hidd_ch),
                nn.ReLU(),
                #nn.Linear(hidd_ch, hidd_ch),
                #nn.ReLU()
            )

        # If using linear after frame stacks here we will calculate the output dimension by multiplying it by frame stacks
        out_shape = self.features(torch.randn(*((1,) + tuple(state_size[:-1])))).view(-1).size().numel()
        self.lstm_out = nn.LSTM(out_shape, hidd_ch, 1, batch_first=True)

        self.advantage = nn.Sequential(
            nn.Linear(hidd_ch, self.action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(hidd_ch, 1)
        )

    def forward(self, obs):
        if obs.ndimension() == 4:
            obs = obs[None]
        obs = obs.float().transpose(2, 4)
        stack = obs.shape[1]
        x = torch.cat([self.features(obs[:, i])[:, None] for i in range(stack)], dim=1)
        x = x.view(x.size(0), stack, -1)

        h0 = torch.zeros(1, x.size(0), self.hidd_ch).to(sett.device)
        c0 = torch.zeros(1, x.size(0), self.hidd_ch).to(sett.device)

        lstm_out, (hn, cn) = self.lstm_out(x, (h0.detach(), c0.detach()))
        x = lstm_out[:, -1]
        adv = self.advantage(x)
        value = self.value(x)
        return value + (adv - adv.mean(-1, keepdim=True))

    def act(self, state, eps):
        if np.random.random() > eps:
            q = self.forward(state)
            action = torch.argmax(q, dim=-1).cpu().data.numpy()
        else:
            action = np.random.randint(self.action_size, size=1 if len(state.shape) == 1 else state.shape[0])
        return action.item() if action.shape == (1,) else list(action.astype(np.int))

    def update_target(self, model):
        self.load_state_dict(model.state_dict())


class ICM_Model(nn.Module):
    def __init__(self, state_size, action_size, conv):
        super(ICM_Model, self).__init__()

        self.action_size = action_size
        self.state_size = np.array(state_size)[[3, 1, 2, 0]]

        # Projection
        if conv:
            self.phi = nn.Sequential(
                nn.Conv2d(self.state_size[0], 32, kernel_size=3, stride=3, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU()
            )
        else:
            self.phi = nn.Sequential(
                nn.Linear(self.state_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Relu()
            )

        out_shape = self.phi(torch.randn(*((1,) + tuple(self.state_size[:-1])))).view(-1).size().numel()

        # Forward Model
        self.fwd = nn.Sequential(
            nn.Linear(out_shape + 1, 256),
            nn.ReLU(),
            nn.Linear(256, out_shape)
        )

        # Inverse Model
        self.inv = nn.Sequential(
            nn.Linear(out_shape * 2, 256),
            nn.ELU(),
            nn.Linear(256, action_size)
        )

    def forward(self, *input):
        obs, action = input
        action = action.view(-1, 1)
        phi = self.phi_state(obs)
        x = torch.cat((phi, action.float()), -1)
        phi_hat = self.fwd(x)
        return phi_hat

    def phi_state(self, s):
        s = s[:, -1]
        x = s.float().transpose(1, 3)
        x = self.phi(x)
        return x.view(x.size(0), -1)

    def inverse_pred(self, s, s1):
        s = self.phi_state(s.float())
        s1 = self.phi_state(s1.float())
        x = torch.cat((s, s1), -1)
        return self.inv(x)

    def curiosity_rew(self, s, s1, a):
        phi_hat = self.forward(s, a)
        phi_s1 = self.phi_state(s1)
        cur_rew = 1 / 2 * (torch.norm(phi_hat - phi_s1, p=2, dim=-1) ** 2)
        return cur_rew

