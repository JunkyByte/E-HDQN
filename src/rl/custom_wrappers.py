import gym
import numpy as np
from gym.spaces import Box
import cv2


class TimeLimit(gym.Wrapper):
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
        return observation, reward, done, None

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class FixGrayScale(gym.Wrapper):
    def __init__(self, env):
        super(FixGrayScale, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if isinstance(observation, np.ndarray) and observation.ndim == 3:
            observation = observation[..., np.newaxis]
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if isinstance(observation, np.ndarray) and observation.ndim == 3:
            observation = observation[..., np.newaxis]
        return observation

class RepeatAction(gym.Wrapper):
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

class ResizeState(gym.Wrapper):
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
        observation = cv2.resize(observation, self.res)
        if self.gray:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = cv2.resize(observation, self.res)
        if self.gray:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation

class LifeLimitMario(gym.Wrapper):
    def __init__(self, env):
        super(LifeLimitMario, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if info['life'] != 2:
            done = True
        return observation, reward, done, info

class TimeLimitMario(gym.Wrapper):
    def __init__(self, env, time):
        super(TimeLimitMario, self).__init__(env)
        self.time = time

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if info['time'] < self.time:
            done = True
        return observation, reward, done, info

class ChannelsConcat(gym.Wrapper):
    def __init__(self, env):
        super(ChannelsConcat, self).__init__(env)
        shape = self.observation_space.shape
        self.shape = (shape[1], shape[2], shape[0] * shape[-1])
        low = self.observation_space.low.reshape(self.shape)
        high = self.observation_space.high.reshape(self.shape)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.array(observation)
        return observation.reshape(self.shape), reward, done, info

    def reset(self, **kwargs):
        observation = np.array(self.env.reset(**kwargs))
        return observation.reshape(self.shape)
