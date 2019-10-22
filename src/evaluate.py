import time
import gym
import Custom_Env.DimensionalGrid
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from rl.ehdqn import EHDQN
import logging
import argparse
import numpy as np
from gym.wrappers import AtariPreprocessing
from rl.frame_stack import FrameStack
from rl.custom_wrappers import FixGrayScale, TimeLimit, TimeLimitMario, RepeatAction, ResizeState, ChannelsConcat, LifeLimitMario
parser = argparse.ArgumentParser()
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser.add_argument('--env', default='DimGrid-v0', type=str, help='Environment name')
parser.add_argument('--ckpt', default='./', type=str, help='Ckpt path')
parser.add_argument('--step', default=-1, type=int, help='Ckpt step value')
parser.add_argument('--episodes', default=1_000_000, type=int, help='number of episodes')
parser.add_argument('--gamma', default=0.9, type=float, help='Discount reward factor')
parser.add_argument('--gamma_macro', default=0.9, type=float, help='Discount reward factor for macro policy')
parser.add_argument('--hidd_ch', default=128, type=int, help='Number of hidden units per hidden channels')
parser.add_argument('--lam', default=0.1, type=float, help='Scaler for intrinsic reward')
parser.add_argument('--embed_state_size', default=128, type=int, help='Number of units for embed representation')
parser.add_argument('--max_time', default=4, type=int, help='Number of steps per policy')
parser.add_argument('--n_subpolicy', default=2, type=int, help='Number of sub policies')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for agent training')
parser.add_argument('--eps', default=0.5, type=float, help='Chance of taking random action')
parser.add_argument('--eps_decay', default=1e-4, type=float, help='Decay for macro eps')
parser.add_argument('--eps_sub', default=0.5, type=float, help='Chance of taking random action for sub policy')
parser.add_argument('--eps_sub_decay', default=1e-4, type=float, help='Decay for sub policy eps')
parser.add_argument('--bs', default=32, type=int, help='Batch size')
parser.add_argument('--train_interval', default=5, type=int, help='Steps of env before training')
parser.add_argument('--train_steps', default=1, type=int, help='Steps of training')
parser.add_argument('--target_int', default=100, type=int, help='Steps of training to update target network in dueling arch.')
parser.add_argument('--max_memory', default=50_000, type=int, help='Max memory')
parser.add_argument('--size', default=20, type=int, help='Size for DimGrid environment')
parser.add_argument('--tau', default=0.01, type=float, help='Weight for agent loss')
parser.add_argument('--beta', default=0.2, type=float, help='Weight for fwd vs inv icm loss')
parser.add_argument('--reward_rescale', default=0., type=float, help='Reward rescaling: 0 -> Id, 1 -> sign(R), float -> Rew * mult')


if __name__ == '__main__':
    args = parser.parse_args()

    # Setup env
    if 'DimGrid' in args.env:
        env = gym.make(args.env, size=args.size)
    elif 'Mario' in args.env:
        env = gym.make(args.env)
        env = env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = RepeatAction(env, nskip=6)
        env = TimeLimitMario(env, time=300) # 400 - time = total_seconds
        #env = LifeLimitMario(env)
        env = ResizeState(env, res=(48, 48), gray=False)
        #env = FixGrayScale(env)
        env = FrameStack(env, num_stack=4)
        env = ChannelsConcat(env)

    obs = env.reset()

    # Setup Model
    n_actions = env.action_space.n if env.action_space.shape == () else env.action_space.shape[0]
    n_state = env.observation_space.n if env.observation_space.shape == () else env.observation_space.shape

    conv = True if isinstance(n_state, tuple) else False
    dqn = EHDQN(state_dim=n_state,
                embed_state_dim=args.embed_state_size,
                tau=args.tau,
                action_dim=n_actions,
                gamma=args.gamma,
                n_subpolicy=args.n_subpolicy,
                max_time=args.max_time,
                hidd_ch=args.hidd_ch,
                lam=args.lam,
                lr=args.lr,
                eps=args.eps,
                eps_decay=args.eps_decay,
                eps_sub=args.eps_sub,
                eps_sub_decay=args.eps_sub_decay,
                beta=args.beta,
                bs=args.bs,
                target_interval=args.target_int,
                train_steps=args.train_steps,
                max_memory=args.max_memory,
                conv=conv,
                reward_rescale=args.reward_rescale,
                gamma_macro=args.gamma_macro
                )

    # Load model
    dqn.load(args.ckpt, i=args.step)

    while True:
        tot_reward = 0
        obs = env.reset()
        while True:
            action = dqn.act(obs[np.newaxis], deterministic=True)
            obs_new, r, is_terminal, _ = env.step(action)

            env.render()
            time.sleep(1e-2)
            tot_reward += r
            obs = obs_new

            if is_terminal:
                break

        logging.info('Reward: %s' % (tot_reward))
