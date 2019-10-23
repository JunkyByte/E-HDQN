import time
from rl.ehdqn import EHDQN
import logging
import argparse
import numpy as np
from environment_manager import create_environment
import parser
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


if __name__ == '__main__':
    args = parser.args.parse_args()

    # Setup env
    env = create_environment(args.env, size=args.size)

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
                max_memory_sub=args.max_memory_sub,
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
            logging.info('Policy: %s Action %s' % (dqn.selected_policy, action))
            obs_new, r, is_terminal, _ = env.step(action)

            env.render()
            time.sleep(1e-2)

            tot_reward += r
            obs = obs_new

            if is_terminal:
                break

        logging.info('Reward: %s' % (tot_reward))
        time.sleep(1)
