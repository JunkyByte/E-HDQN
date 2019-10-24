import numpy as np
import torch
from src.rl.model import DDQN_Model, ICM_Model
from src.rl.memory import Memory
from torch.nn.functional import mse_loss, cross_entropy, smooth_l1_loss
import settings as sett
import itertools
import os


class EHDQN:
    def __init__(self, state_dim, embed_state_dim, tau, action_dim, gamma, n_subpolicy, max_time, hidd_ch, lam, lr, eps,
                 eps_decay, eps_sub, eps_sub_decay, beta, bs, target_interval, train_steps, max_memory, max_memory_sub,
                 conv, gamma_macro, reward_rescale, norm_input=True, logger=None):
        """
        :param state_dim: Shape of the state
        :param int embed_state_dim: Length of the embed state
        :param float tau: Weight for agent loss
        :param gamma_macro: Discount for macro controller
        :param int action_dim: Number of actions
        :param float gamma: Discount for sub controller
        :param int n_subpolicy: Number of sub policies
        :param int max_time: Number of steps for each sub policy
        :param int hidd_ch: Number of hidden channels
        :param float lam: Scaler for ICM reward
        :param float lr: Learning rate
        :param float eps: Eps greedy chance for macro policy
        :param float eps_decay: Epsilon decay computed as eps * (1 - eps_decay) each step
        :param float eps_sub: Eps greedy change for sub policies
        :param float eps_sub_decay: Epsilon decay for sub policy computed as eps * (1 - eps_decay) each step
        :param float beta: Weight for loss of fwd net vs inv net
        :param int bs: Batch size
        :param int target_interval: Number of train steps between target updates
        :param int train_steps: Number of training iterations for each call
        :param int max_memory: Max memory
        :param bool conv: Use or not convolutional networks
        :param int max_time: Maximum steps for sub policy
        """

        # Parameters
        self.logger = logger
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_interval = target_interval
        self.lr = lr
        self.bs = bs
        # Macro Policy parameters
        self.eps = eps
        self.eps_decay = 1 - eps_decay
        self.gamma_macro = gamma_macro
        # Sub policy parameters
        self.n_subpolicy = n_subpolicy
        self.tau = tau
        self.eps_sub = eps_sub
        self.eps_sub_decay = 1 - eps_sub_decay
        self.gamma = gamma
        # ICM parameters
        self.embed_state_dim = embed_state_dim
        self.beta = beta
        self.lam = lam

        self.selected_policy = None
        self.macro_state = None
        self.max_time = max_time
        self.train_steps = train_steps
        self.reward_rescale = reward_rescale
        self.norm_input = norm_input
        self.curr_time = 0
        self.macro_reward = 0
        self.target_count = np.zeros((self.n_subpolicy,), dtype=np.int)
        self.counter_macro = np.zeros((self.n_subpolicy,), dtype=np.int)
        self.macro_count = 0

        # Create Policies / ICM modules / Memories
        self.macro = DDQN_Model(state_dim, n_subpolicy, hidd_ch).to(sett.device)
        self.macro_target = DDQN_Model(state_dim, n_subpolicy, hidd_ch).to(sett.device)
        self.macro_target.update_target(self.macro)
        self.macro_memory = Memory(max_memory)
        self.macro_opt = torch.optim.Adam(self.macro.parameters(), lr=self.lr)
        self.memory, self.policy, self.target, self.icm, self.policy_opt = [], [], [], [], []
        for i in range(n_subpolicy):
            # Create sub-policies
            self.policy.append(DDQN_Model(state_dim, action_dim, conv, hidd_ch).to(sett.device))
            self.target.append(DDQN_Model(state_dim, action_dim, conv, hidd_ch).to(sett.device))
            self.target[-1].update_target(self.policy[-1])
            self.memory.append(Memory(max_memory_sub))

            # Create ICM modules
            self.icm.append(ICM_Model(self.state_dim, self.embed_state_dim, self.action_dim, conv, hidd_ch).to(sett.device))

            # Create sub optimizers
            params = [self.icm[i].parameters(), self.policy[i].parameters()]
            self.policy_opt.append(torch.optim.Adam(itertools.chain(*params), lr=self.lr))

    def save(self, i):
        torch.save(self.macro.state_dict(), os.path.join(sett.SAVEPATH, 'Macro_%s.pth' % i))
        for sub in range(self.n_subpolicy):
            torch.save(self.policy[sub].state_dict(), os.path.join(sett.SAVEPATH, 'Sub_%s_%s.pth' % (sub, i)))

    def load(self, path, i):
        self.macro.load_state_dict(torch.load(os.path.join(path, 'Macro_%s.pth' % i), map_location=sett.device))
        for sub in range(self.n_subpolicy):
            self.policy[sub].load_state_dict(torch.load(os.path.join(path, 'Sub_%s_%s.pth' % (sub, i)), map_location=sett.device))

    def act(self, obs, deterministic=False):
        x = torch.from_numpy(obs).float().to(sett.device)
        if self.norm_input:
            x /= 255

        if self.selected_policy is None or self.curr_time == self.max_time:
            if not self.selected_policy is None and not deterministic:
                # Store non terminal macro transition
                self.macro_reward /= self.max_time
                self.macro_memory.store_transition(self.macro_state, obs[0], self.selected_policy, self.macro_reward, False)
                self.macro_reward = 0

            # Pick macro action
            self.selected_policy = self.pick_policy(x, deterministic=deterministic)
            if not deterministic:
                self.macro_state = obs[0]

            self.counter_macro[self.selected_policy] += 1

        eps = max(0.01, self.eps_sub) if not deterministic else 0.01
        action = self.policy[self.selected_policy].act(x, eps=eps)
        self.curr_time += 1
        return action

    def pick_policy(self, obs, deterministic=False):
        eps = max(0.01, self.eps) if not deterministic else 0.01
        cat = True if not deterministic else False

        policy = self.macro.act(obs, eps=eps, categorical=cat)
        self.curr_time = 0
        return policy

    def set_mode(self, training=False):
        for policy in self.policy:
            policy.train(training)
        self.macro.train(training)

    def store_transition(self, s, s1, a, reward, is_terminal):
        # Rescale reward if a scaling is provided
        if self.reward_rescale != 0:
            if self.reward_rescale == 1:
                reward = np.sign(reward)
            elif self.reward_rescale == 2:
                reward = max(-1, min(1, reward))
            else:
                reward *= self.reward_rescale

        # Store sub policy experience
        self.memory[self.selected_policy].store_transition(s, s1, a, reward, is_terminal)
        self.macro_reward += reward

        # Store terminal macro transition
        if is_terminal:
            self.macro_reward /= self.max_time
            self.macro_memory.store_transition(self.macro_state, s1, self.selected_policy, self.macro_reward, is_terminal)
            self.macro_reward = 0
            if is_terminal:
                self.selected_policy = None

    def update(self):
        for i in range(self.train_steps):
            self._update()
            if self.logger is not None:
                self.logger.step += 1

    def _update(self):
        # First train each sub policy
        for i in range(self.n_subpolicy):
            memory = self.memory[i]
            if len(memory.state) < self.bs * 10:
                continue

            policy = self.policy[i]
            target = self.target[i]
            icm = self.icm[i]
            policy_opt = self.policy_opt[i]

            state, new_state, action, reward, is_terminal = memory.sample(self.bs)
            if self.norm_input:
                state = np.array(state, dtype=np.float) / 255
                new_state = np.array(new_state, dtype=np.float) / 255

            state = torch.tensor(state, dtype=torch.float).detach().to(sett.device)
            new_state = torch.tensor(new_state, dtype=torch.float).detach().to(sett.device)
            action = torch.tensor(action).detach().to(sett.device)
            reward = torch.tensor(reward, dtype=torch.float).detach().to(sett.device)
            is_terminal = torch.tensor(is_terminal, dtype=torch.float).detach().to(sett.device)

            # Augment rewards with curiosity
            curiosity_rewards = icm.curiosity_rew(state, new_state, action)
            reward += self.lam * curiosity_rewards

            # Policy loss
            q = policy(state)[torch.arange(self.bs), action]
            max_action = torch.argmax(policy(new_state), dim=1)
            y = reward + self.gamma * target(new_state)[torch.arange(self.bs), max_action] * is_terminal
            policy_loss = smooth_l1_loss(input=q, target=y.detach())

            # ICM Loss
            phi_hat = icm(state, action)
            phi_true = icm.phi_state(new_state)
            fwd_loss = mse_loss(input=phi_hat, target=phi_true.detach())
            a_hat = icm.inverse_pred(state, new_state)
            inv_loss = cross_entropy(input=a_hat, target=action)

            # Total loss
            loss = self.tau * policy_loss + (1 - self.beta) * inv_loss + self.beta * fwd_loss
            policy_opt.zero_grad()
            loss.backward()
            for param in policy.parameters():
                param.grad.data.clamp(-1, 1)
            policy_opt.step()

            self.target_count[i] += 1
            if self.target_count[i] == self.target_interval:
                self.target_count[i] = 0
                self.target[i].update_target(self.policy[i])

            if self.logger is not None:
                self.logger.log_scalar(tag='Policy Loss %i' % i, value=policy_loss.cpu().detach().numpy())
                self.logger.log_scalar(tag='ICM Fwd Loss %i' % i, value=fwd_loss.cpu().detach().numpy())
                self.logger.log_scalar(tag='ICM Inv Loss %i' % i, value=inv_loss.cpu().detach().numpy())
                self.logger.log_scalar(tag='Total Policy Loss %i' % i, value=loss.cpu().detach().numpy())
                self.logger.log_scalar(tag='Mean Curiosity Reward %i' % i, value=curiosity_rewards.mean().cpu().detach().numpy())

        # Reduce sub eps
        self.eps_sub = self.eps_sub * self.eps_sub_decay

        # Train Macro policy
        if len(self.macro_memory.state) < self.bs * 10:
            return

        # Reduce eps
        self.eps = self.eps * self.eps_decay

        state, new_state, action, reward, is_terminal = self.macro_memory.sample(self.bs)
        if self.norm_input:
            state = np.array(state, dtype=np.float) / 255
            new_state = np.array(new_state, dtype=np.float) / 255

        state = torch.tensor(state, dtype=torch.float).detach().to(sett.device)
        new_state = torch.tensor(new_state, dtype=torch.float).detach().to(sett.device)
        action = torch.tensor(action).detach().to(sett.device)
        reward = torch.tensor(reward, dtype=torch.float).detach().to(sett.device)
        is_terminal = torch.tensor(is_terminal, dtype=torch.float).detach().to(sett.device)

        q = self.macro(state)[torch.arange(self.bs), action]
        max_action = torch.argmax(self.macro(new_state), dim=1)
        y = reward + self.gamma_macro * self.macro_target(new_state)[torch.arange(self.bs), max_action] * is_terminal
        loss = smooth_l1_loss(input=q, target=y.detach())

        self.macro_opt.zero_grad()
        loss.backward()
        for param in self.macro.parameters():
            param.grad.data.clamp(-1, 1)
        self.macro_opt.step()

        self.macro_count += 1
        if self.macro_count == self.target_interval:
            self.macro_count = 0
            self.macro_target.update_target(self.macro)

        if self.logger is not None:
            self.logger.log_scalar(tag='Macro Loss', value=loss.cpu().detach().numpy())
            self.logger.log_scalar(tag='Sub Eps', value=self.eps_sub)
            self.logger.log_scalar(tag='Macro Eps', value=self.eps)
            values = self.counter_macro / max(1, np.sum(self.counter_macro))
            self.logger.log_text(tag='Macro Policy Actions', value=[str(v) for v in values],
                                 step=self.logger.step)
            self.logger.log_histogram(tag='Macro Policy Actions0', values=values,
                                      step=self.logger.step, bins=2)
