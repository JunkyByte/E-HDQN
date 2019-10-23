import gym
import Custom_Env.DimensionalGrid
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import AtariPreprocessing
from rl.frame_stack import FrameStack
from rl.custom_wrappers import FixGrayScale, TimeLimit, TimeLimitMario, RepeatAction, ResizeState, ChannelsConcat, LifeLimitMario


def create_environment(env_name, **kwargs):
    if 'DimGrid' in env_name:
        env = gym.make(env_name, size=kwargs['size'])
    elif 'Mario' in env_name:
        env = gym.make(env_name)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = RepeatAction(env, nskip=6)
        # env = TimeLimitMario(env, time=300) # 400 - time = total_seconds
        # env = LifeLimitMario(env)
        env = ResizeState(env, res=(84, 84), gray=True)
        env = FixGrayScale(env)
        env = FrameStack(env, num_stack=4)
        env = ChannelsConcat(env)

    env.seed(42)
    return env
