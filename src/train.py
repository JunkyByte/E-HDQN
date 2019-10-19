import gym
import Custom_Env.DimensionalGrid
from rl.ehdqn import EHDQN
import logging
import argparse
import logger as log
import numpy as np
from gym.wrappers import AtariPreprocessing
parser = argparse.ArgumentParser()
logging.basicConfig(format='%(asctime)s;%(levelname)s:%(message)s', level=logging.INFO)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser.add_argument('--env', default='DimGrid-v0', type=str, help='Environment name')
parser.add_argument('--episodes', default=1_000_000, type=int, help='number of episodes')
parser.add_argument('--gamma', default=0.9, type=float, help='Discount reward factor')
parser.add_argument('--gamma_macro', default=0.9, type=float, help='Discount reward factor for macro policy')
parser.add_argument('--hidd_ch', default=128, type=int, help='Number of hidden units per hidden channels')
parser.add_argument('--lam', default=0.01, type=float, help='Scaler for intrinsic reward')
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
parser.add_argument('--max_memory', default=100_000, type=int, help='Max memory')
parser.add_argument('--size', default=20, type=int, help='Size for DimGrid environment')
parser.add_argument('--tau', default=0.01, type=float, help='Weight for agent loss')
parser.add_argument('--beta', default=0.2, type=float, help='Weight for fwd vs inv icm loss')


if __name__ == '__main__':
    args = parser.parse_args()

    # Setup env
    env = gym.make(args.env, size=args.size)
    env = Custom_Env.DimensionalGrid.TimeLimit(env, args.size * 3)
    #env = AtariPreprocessing(env, noop_max=30, screen_size=64,
    #                         terminal_on_life_loss=True, grayscale_obs=False)
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
                gamma_macro=args.gamma_macro,
                logger=log.TB_LOGGER
                )

    train_steps = 0
    tot_succ = 0
    for i in range(args.episodes):
        log.TB_LOGGER.step += 1
        obs = env.reset()

        while True:
            action = dqn.act(obs[np.newaxis])
            obs_new, r, is_terminal, _ = env.step(action)

            tot_succ += r
            dqn.store_transition(obs, obs_new, action, r, is_terminal)

            train_steps += 1
            if train_steps % args.train_interval == 0 and train_steps > 0:
                train_steps = 0
                dqn.update()
                log.TB_LOGGER.step += 1

            obs = obs_new
            if is_terminal:
                break

        episodes_per_epoch = 100
        if i % episodes_per_epoch == 0 and i > 0:
            log.TB_LOGGER.log_scalar(tag='Train Reward:', value=tot_succ / episodes_per_epoch)
            tot_succ = 0

        if i % 1000 == 0:
            n_eval_episodes = 100
            tot_reward = 0
            for _ in range(n_eval_episodes):
                obs = env.reset()
                while True:
                    action = dqn.act(obs[np.newaxis], deterministic=True)
                    obs_new, r, is_terminal, _ = env.step(action)

                    #env.render()
                    tot_reward += r
                    obs = obs_new

                    if is_terminal:
                        break

            eval_succ = tot_reward / n_eval_episodes
            logging.info('Mean Reward: %s' % (eval_succ))
            log.TB_LOGGER.log_scalar(tag='Eval Reward', value=eval_succ)
