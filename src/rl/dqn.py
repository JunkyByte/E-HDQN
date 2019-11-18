import numpy as np
import torch
from rl.model import DDQN_Model, ICM_Model
from rl.memory import Memory
from rl.per import PERMemory
from torch.nn.functional import mse_loss, cross_entropy, smooth_l1_loss, softmax
import settings as sett
import itertools
import os


class DQN:
    def __init__(self, state_dim, tau, action_dim, gamma, hidd_ch, lam, lr,
                 eps_sub, eps_sub_decay, beta, bs, target_interval, train_steps, max_memory,
                 conv, reward_rescale, n_proc, per=False, norm_input=True, logger=None):
        """
        :param state_dim: Shape of the state
        :param float tau: Weight for agent loss
        :param int action_dim: Number of actions
        :param float gamma: Discount for sub controller
        :param int hidd_ch: Number of hidden channels
        :param float lam: Scaler for ICM reward
        :param float lr: Learning rate
        :param float eps_sub: Eps greedy change for sub policies
        :param float eps_sub_decay: Epsilon decay for sub policy computed as eps * (1 - eps_decay) each step
        :param float beta: Weight for loss of fwd net vs inv net
        :param int bs: Batch size
        :param int target_interval: Number of train steps between target updates
        :param int train_steps: Number of training iterations for each call
        :param int max_memory: Max memory
        :param bool conv: Use or not convolutional networks
        :param bool per: Use or not prioritized experience replay
        """

        # Parameters
        self.logger = logger
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_interval = target_interval
        self.lr = lr
        self.bs = bs
        # policy parameters
        self.tau = tau
        self.eps_sub = eps_sub
        self.eps_sub_decay = 1 - eps_sub_decay
        self.gamma = gamma
        # ICM parameters
        self.beta = beta
        self.lam = lam

        self.n_proc = n_proc
        self.train_steps = train_steps
        self.reward_rescale = reward_rescale
        self.norm_input = norm_input
        self.per = per
        self.target_count = 0

        if self.per:
            memory = PERMemory
        else:
            memory = Memory

        # Create Policies / ICM modules / Memories
        self.agent = DDQN_Model(self.state_dim, self.action_dim, hidd_ch)
        self.agent_target = DDQN_Model(self.state_dim, self.action_dim, hidd_ch)
        self.agent_target.update_target(self.agent)
        self.agent_memory = memory(max_memory)
        self.agent_opt = torch.optim.Adam(self.agent.parameters(), lr=self.lr)
        self.icm = ICM_Model(self.state_dim, self.action_dim, conv).to(sett.device)
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=1e-3)

        # Send macro to correct device
        self.agent = self.agent.to(sett.device)
        self.agent_target = self.agent_target.to(sett.device)

    def act(self, obs, deterministic=False):
        x = torch.from_numpy(obs).float().to(sett.device)
        if self.norm_input:
            x /= 255

        eps = max(0.01, self.eps_sub) if not deterministic else 0.01
        actions = self.agent.act(x, eps=eps)
        return actions

    def set_mode(self, training=False):
        self.agent.train(training)

    def process_reward(self, reward):
        # Rescale reward if a scaling is provided
        if self.reward_rescale != 0:
            if self.reward_rescale == 1:
                reward = np.sign(reward)
            elif self.reward_rescale == 2:
                reward = np.clip(reward, -1, 1)
            else:
                reward *= self.reward_rescale
        return reward

    def store_transition(self, s, s1, a, reward, is_terminal):
        reward = self.process_reward(reward)
        for i in range(len(s)):
            self.agent_memory.store_transition(s[i], s1[i], a[i], reward[i], is_terminal[i])

    def update(self):
        for i in range(self.train_steps):
            self._update()
            if self.logger is not None:
                self.logger.step += 1

    def _update(self):
        # First train each sub policy
        i = 0
        memory = self.agent_memory
        if len(memory) < self.bs * 10:
            return

        policy = self.agent
        target = self.agent_target
        icm = self.icm
        policy_opt = self.agent_opt
        icm_opt = self.icm_opt

        if self.per:
            state, new_state, action, reward, is_terminal, idxs, w_is = memory.sample(self.bs)
            reduction = 'none'
            self.logger.log_scalar(tag='Beta PER %i' % i, value=memory.beta)
        else:
            state, new_state, action, reward, is_terminal = memory.sample(self.bs)
            reduction = 'mean'

        if self.norm_input:
            state = np.array(state, dtype=np.float) / 255
            new_state = np.array(new_state, dtype=np.float) / 255

        state = torch.tensor(state, dtype=torch.float).detach().to(sett.device)
        new_state = torch.tensor(new_state, dtype=torch.float).detach().to(sett.device)
        action = torch.tensor(action).detach().to(sett.device)
        reward = torch.tensor(reward, dtype=torch.float).detach().to(sett.device)
        is_terminal = 1. - torch.tensor(is_terminal, dtype=torch.float).detach().to(sett.device)

        # Augment rewards with curiosity
        curiosity_rewards = icm.curiosity_rew(state, new_state, action)
        reward = (1 - 0.01) * reward + 0.01 * self.lam * curiosity_rewards

        # Policy loss
        q = policy.forward(state, macro=self.macro)[torch.arange(self.bs), action]
        max_action = torch.argmax(policy.forward(new_state, macro=self.macro), dim=1)
        y = reward + self.gamma * target.forward(new_state, macro=self.macro)[torch.arange(self.bs), max_action] * is_terminal
        policy_loss = smooth_l1_loss(input=q, target=y.detach(), reduction=reduction).mean(-1)

        # ICM Loss
        phi_hat = icm.forward(state, action)
        phi_true = icm.phi_state(new_state)
        fwd_loss = mse_loss(input=phi_hat, target=phi_true.detach(), reduction=reduction).mean(-1)
        a_hat = icm.inverse_pred(state, new_state)
        inv_loss = cross_entropy(input=a_hat, target=action.detach(), reduction=reduction)

        # Total loss
        inv_loss = (1 - self.beta) * inv_loss
        fwd_loss = self.beta * fwd_loss * 288
        loss = self.tau * policy_loss + inv_loss + fwd_loss

        if self.per:
            error = np.clip((torch.abs(q - y)).cpu().data.numpy(), 0, 0.8) # TODO
            inv_prob = (1 - softmax(a_hat, dim=1)[torch.arange(self.bs), action]) / 5
            curiosity_error = torch.abs(inv_prob).cpu().data.numpy()
            total_error = error + curiosity_error

            # update priorities
            for k in range(self.bs):
                memory.update(idxs[k], total_error[k])

            loss = (loss * torch.FloatTensor(w_is).to(sett.device)).mean()

        policy_opt.zero_grad()
        icm_opt.zero_grad()
        loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp(-1, 1)
        policy_opt.step()
        icm_opt.step()

        self.target_count += 1
        if self.target_count == self.target_interval:
            self.target_count = 0
            self.agent_target.update_target(self.agent)

        if self.logger is not None:
            self.logger.log_scalar(tag='Policy Loss %i' % i, value=policy_loss.mean().cpu().data.numpy())
            self.logger.log_scalar(tag='ICM Fwd Loss %i' % i, value=fwd_loss.mean().cpu().data.numpy())
            self.logger.log_scalar(tag='ICM Inv Loss %i' % i, value=inv_loss.mean().cpu().data.numpy())
            self.logger.log_scalar(tag='Total Policy Loss %i' % i, value=loss.mean().cpu().data.numpy())
            self.logger.log_scalar(tag='Mean Curiosity Reward %i' % i, value=curiosity_rewards.mean().cpu().data.numpy())
            self.logger.log_scalar(tag='Q values %i' % i, value=q.mean().cpu().data.numpy())
            self.logger.log_scalar(tag='Target Boltz %i' % i, value=y.mean().cpu().data.numpy())
            if self.per:
                self.logger.log_scalar(tag='PER Error %i' % i, value=total_error.mean())
                self.logger.log_scalar(tag='PER Error Policy %i' % i, value=error.mean())
                self.logger.log_scalar(tag='PER Error Curiosity %i' % i, value=curiosity_error.mean())

        # Reduce sub eps
        self.eps_sub = self.eps_sub * self.eps_sub_decay
