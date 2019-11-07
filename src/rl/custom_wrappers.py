import numpy as np
import cv2
from gym import Wrapper
from gym.spaces import Box


class TimeLimit(Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None:
            max_episode_steps = env.spec.max_episode_steps
        self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class FixGrayScale(Wrapper):
    def __init__(self, env):
        super(FixGrayScale, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if isinstance(observation, np.ndarray) and observation.ndim == 2:
            observation = observation[..., np.newaxis]
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if isinstance(observation, np.ndarray) and observation.ndim == 2:
            observation = observation[..., np.newaxis]
        return observation

class RepeatAction(Wrapper):
    def __init__(self, env, nskip=4):
        super(RepeatAction, self).__init__(env)
        self.nskip = nskip

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        for i in range(self.nskip - 1):
            if not done:
                observation, r, done, info = self.env.step(action)
                reward += r
        return observation, reward, done, info

class ResizeState(Wrapper):
    def __init__(self, env, res=(64, 64), gray=False):
        super(ResizeState, self).__init__(env)
        self.res = res
        self.gray = gray

        low = self.observation_space.low[:res[0], :res[1], :]
        high = self.observation_space.high[:res[0], :res[1], :]
        if gray:
            low = low[..., 0:1]
            high = high[..., 0:1]
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = cv2.resize(observation, self.res, interpolation=cv2.INTER_AREA)
        observation = cv2.convertScaleAbs(observation, alpha=1.25, beta=25)
        if self.gray:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = cv2.resize(observation, self.res, interpolation=cv2.INTER_AREA)
        observation = cv2.convertScaleAbs(observation, alpha=1.25, beta=25)
        if self.gray:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation

class LifeLimitMario(Wrapper):
    def __init__(self, env):
        super(LifeLimitMario, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if info['life'] != 2:
            done = True
        return observation, reward, done, info

class TimeLimitMario(Wrapper):
    def __init__(self, env, time):
        super(TimeLimitMario, self).__init__(env)
        self.time = time

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if info['time'] < self.time:
            done = True
        return observation, reward, done, info

class ChannelsConcat(Wrapper):
    def __init__(self, env):
        super(ChannelsConcat, self).__init__(env)
        shape = self.observation_space.shape
        self.shape = (shape[1], shape[2], shape[0] * shape[-1])
        low = self.observation_space.low.reshape(self.shape)
        high = self.observation_space.high.reshape(self.shape)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def concat(self, x):
        return np.concatenate(x, axis=-1)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.array(observation)
        return self.concat(observation), reward, done, info

    def reset(self, **kwargs):
        observation = np.array(self.env.reset(**kwargs))
        return self.concat(observation)

class RewardSparse(Wrapper):
    def __init__(self, env, very_sparse=False):
        super(RewardSparse, self).__init__(env)
        self.very_sparse = very_sparse
        self.max_pos = 0
        self.max_time = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = 0
        if not done:
            if not self.very_sparse and info['x_pos'] - self.max_pos > 200:  # Going forward
                reward = 1
                self.max_pos = info['x_pos']
                self.max_time = info['time']
            elif abs(info['x_pos'] - self.max_pos) < 50 and self.max_time - info['time'] > 10:
                reward = -1
                done = True
        else:
            reward = -1  # End of episode

        return observation, reward, done, info

    def reset(self, **kwargs):
        self.max_pos = 0
        return self.env.reset(**kwargs)
