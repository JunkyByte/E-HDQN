import gym
import Custom_Env.DimensionalGrid
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import AtariPreprocessing
from rl.frame_stack import FrameStack
from rl.custom_wrappers import FixGrayScale, TimeLimit, TimeLimitMario, RepeatAction, ResizeState, ChannelsConcat, LifeLimitMario, RewardSparse
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


def create_environment(env_name, n_env, **kwargs):
    env = SubprocVecEnv([make_env(env_name, i, **kwargs) for i in range(n_env)])
    return env

def make_env(env_id, rank, seed=0, **kwargs):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        if 'DimGrid' in env_id:
            env = gym.make(env_id, size=kwargs['size'])
            env = TimeLimit(env, max_episode_steps=kwargs['size'] * 4)
            env = FrameStack(env, num_stack=1)
        elif 'Mario' in env_id:
            nskip = 6
            if 'nskip' in kwargs.keys():
                nskip = kwargs['nskip']

            env = gym.make(env_id)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = RepeatAction(env, nskip=nskip)
            # env = TimeLimitMario(env, time=300) # 400 - time = total_seconds
            # env = LifeLimitMario(env)
            env = ResizeState(env, res=(84, 84), gray=True)
            env = FixGrayScale(env)
            env = FrameStack(env, num_stack=4)
            #env = ChannelsConcat(env)
            if 'sparse' in kwargs.keys():
                if kwargs['sparse'] == 1:
                    env = RewardSparse(env)
                elif kwargs['sparse'] == 2:
                    env = RewardSparse(env, very_sparse=True)
        else: # assume atari
            env = gym.make(env_id)
            env = AtariPreprocessing(env, terminal_on_life_loss=False, frame_skip=4)
            env = FixGrayScale(env)
            env = FrameStack(env, num_stack=4)
        env.seed(seed + rank)
        return env
    return _init