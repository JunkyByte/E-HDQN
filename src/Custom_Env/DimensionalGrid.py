from gym import Env
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import logging

logger = logging.getLogger(__name__)


class DimGridEnvironment(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size, hard):
        super(DimGridEnvironment, self).__init__()
        self.size = size
        self.observation_space = spaces.Box(0, 1, [self.size, self.size, 1])
        self.action_space = spaces.Discrete(5)
        self.alternate_dim = False
        self.dim0 = None
        self.dim1 = None
        self.dict = {'0': {'empty': 0, 'wall': 0.33, 'player': 0.66, 'goal': 1},
                     '1': {'empty': 1, 'wall': 0.66, 'player': 0.33, 'goal': 0}}
        self.action_dict = {0: [0, -1], 1: [-1, 0], 2: [0, 1], 3: [1, 0], 4: 4}
        self.goal = None
        self.player_pos = None
        self.walls = None
        self.hard = hard
        self.seed = self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def _compute_action(self, action):
        action = self.action_dict[action]
        reward = 0

        if action == 4: # Change dimension
            self.alternate_dim = not self.alternate_dim
        else:
            new_pos = tuple(np.array(self.player_pos) + action)
            if any(np.array(new_pos) == self.size) or any(np.array(new_pos) < 0):
                reward = -1 if not self.hard else 0
                return reward, True
            self.player_pos = new_pos
            touched_wall = self.walls[self.player_pos]
            if self.alternate_dim:
                reward += -0.1
            else:
                if touched_wall:
                    reward += -0.5
                    if self.hard:
                        return 0, True

        terminal = True if self.player_pos == self.goal else False
        reward += int(terminal)
        return reward, terminal

    def step(self, action):
        reward, terminal = self._compute_action(action)
        obs = self._get_obs()
        return obs, reward, terminal, None

    def _get_obs(self):
        if self.alternate_dim:
            self.dim1[self.walls] = self.dict['1']['wall']
            self.dim1[self.player_pos] = self.dict['1']['player']
            return self.dim1
        else:
            self.dim0[self.walls] = self.dict['0']['wall']
            self.dim0[self.player_pos] = self.dict['0']['player']
            return self.dim0

    def reset(self):
        y = np.random.choice(self.size, 2, replace=False)
        x = np.random.choice(self.size, 2, replace=False)
        self.player_pos = tuple([x[0], y[0]])
        self.goal = tuple([x[1], y[1]])

        # Fill dim0
        self.dim0 = np.zeros(self.observation_space.shape)
        self.walls = np.random.rand(*self.observation_space.shape) > 0.5
        self.dim0[self.walls] = self.dict['0']['wall']
        self.dim0[self.goal] = self.dict['0']['goal']
        self.dim0[self.player_pos] = self.dict['0']['player']

        # Fill dim1
        self.dim1 = np.ones(self.observation_space.shape)
        self.dim1[self.walls] = self.dict['1']['wall']
        self.dim1[self.goal] = self.dict['1']['goal']
        self.dim1[self.player_pos] = self.dict['1']['player']

        self.walls[self.goal] = False
        self.walls[self.player_pos] = False
        self.alternate_dim = False
        return self._get_obs()

    def render(self, mode='human', close=False):
        logger.info('\n Dim0: %s \n Dim1:  %s' % (self.dim0, self.dim1))
