import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class DDQN_Model(nn.Module):
    def __init__(self, state_size, action_size, conv, hidd_ch=128):
        super(DDQN_Model, self).__init__()
        self.action_size = action_size
        state_size = np.array(state_size)[[3, 1, 2, 0]]

        if conv:
            self.features = nn.Sequential(
                nn.Conv2d(state_size[0], 32, kernel_size=6, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=6, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(state_size, hidd_ch),
                nn.ReLU(),
                #nn.Linear(hidd_ch, hidd_ch),
                #nn.ReLU()
            )

        # If using linear after frame stacks here we will calculate the output dimension by multiplying it by frame stacks
        out_shape = self.features(torch.randn(*((1,) + tuple(state_size[:-1])))).view(-1).size().numel() * state_size[-1]
        self.advantage = nn.Sequential(
            nn.Linear(out_shape, self.action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(out_shape, 1)
        )

    def forward(self, obs):
        if obs.ndimension() == 4:
            obs = obs[None]
        obs = obs.float().transpose(2, 4)
        stack = obs.shape[1]
        x = torch.cat([self.features(obs[:, i])[:, None] for i in range(stack)], dim=1)
        x = x.view(x.size(0), -1)
        adv = self.advantage(x)
        value = self.value(x)
        return value + (adv - adv.mean(-1, keepdim=True))

    def act(self, state, eps, categorical=False):
        if np.random.random() > eps:
            if categorical:
                q = self.forward(state)
                action = Categorical(logits=q).sample().cpu().data.numpy()
            else:
                q = self.forward(state)
                action = torch.argmax(q, dim=-1).cpu().data.numpy()
        else:
            action = np.random.randint(self.action_size, size=1 if len(state.shape) == 1 else state.shape[0])
        return action.item()

    def update_target(self, model):
        self.load_state_dict(model.state_dict())


class ICM_Model(nn.Module):
    def __init__(self, state_size, embed_state_size, action_size, conv, hidd_ch=128):
        super(ICM_Model, self).__init__()

        self.action_size = action_size
        self.embed_state_size = embed_state_size
        self.state_size = np.array(state_size)[[3, 1, 2, 0]]

        # Forward Model
        self.fwd = nn.Sequential(
            nn.Linear(self.embed_state_size + 1, hidd_ch),
            nn.ReLU(),
            nn.Linear(hidd_ch, self.embed_state_size)
        )

        # Projection
        if conv:
            self.phi = nn.Sequential(
                nn.Conv2d(self.state_size[0], 32, kernel_size=6, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=6, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        else:
            self.phi = nn.Sequential(
                nn.Linear(self.state_size, hidd_ch),
                nn.ReLU(),
                nn.Linear(hidd_ch, hidd_ch),
                nn.ReLU(),
                nn.Linear(hidd_ch, hidd_ch),
                nn.Relu()
            )

        out_shape = self.phi(torch.randn(*((1,) + tuple(self.state_size[:-1])))).view(-1).size().numel() * self.state_size[-1]
        self.proj_out = nn.Linear(out_shape, embed_state_size)

        # Inverse Model
        self.inv = nn.Sequential(
            nn.Linear(2 * embed_state_size, hidd_ch),
            nn.ReLU(),
            nn.Linear(hidd_ch, action_size)
        )

    def forward(self, *input):
        obs, action = input
        action = action.view(-1, 1)
        phi = self.phi_state(obs)
        x = torch.cat((phi, action.float()), -1)
        phi_hat = self.fwd(x)
        return phi_hat

    def phi_state(self, s):
        x = s.float().transpose(2, 4)
        stack = x.shape[1]
        x = torch.cat([self.phi(x[:, i])[:, None] for i in range(stack)], dim=1)
        x = x.view(x.size(0), -1)
        return self.proj_out(x)

    def inverse_pred(self, s, s1):
        s = self.phi_state(s.float())
        s1 = self.phi_state(s1.float())
        x = torch.cat((s, s1), -1)
        return self.inv(x)

    def curiosity_rew(self, s, s1, a):
        phi_hat = self(s, a)
        phi_s1 = self.phi_state(s1)
        cur_rew = 1 / 2 * torch.norm(phi_hat - phi_s1, dim=-1) ** 2
        return cur_rew

