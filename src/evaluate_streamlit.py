import time
from rl.ehdqn import EHDQN
import logging
import argparse
import numpy as np
from environment_manager import create_environment
import parser
import streamlit as st
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
st.title('Environment Evaluation')


if __name__ == '__main__':
    args = parser.args.parse_args()
    image = st.empty()

    nskip = 6
    is_mario = True if 'Mario' in args.env and 'Random' not in args.env else False
    norm_input = True if is_mario else False
    if is_mario:
        world = st.selectbox('World', list(range(1, 9)))
        env = 'SuperMarioBros-%s-1-v1' % world
        nskip = st.selectbox('NSkip', [6] + list(range(1, 12)))

    # Setup env
    env = create_environment(env, n_env=1, seed=42, size=args.size, nskip=nskip, sparse=args.sparse)

    obs = env.reset()

    # Setup Model
    n_actions = env.action_space.n if env.action_space.shape == () else env.action_space.shape[0]
    n_state = env.observation_space.n if env.observation_space.shape == () else env.observation_space.shape

    conv = True if isinstance(n_state, tuple) else False
    dqn = EHDQN(state_dim=n_state,
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
                n_proc=1,
                target_interval=args.target_int,
                train_steps=args.train_steps,
                max_memory=args.max_memory,
                max_memory_sub=args.max_memory_sub,
                conv=conv,
                reward_rescale=args.reward_rescale,
                gamma_macro=args.gamma_macro,
                norm_input=norm_input
                )

    # Load model
    dqn.load(args.ckpt, i=args.step)
    dqn.set_mode(training=False)

    while True:
        tot_reward = 0
        obs = env.reset()
        while True:
            action = dqn.act(obs, deterministic=True)
            logging.info('Policy: %s Action %s' % (dqn.selected_policy, action))
            obs_new, r, is_terminal, _ = env.step(action)

            img = env.render(mode='rgb_array')
            image.image(img)

            tot_reward += r[0]
            obs = obs_new

            if is_terminal:
                break

        logging.info('Reward: %s' % (tot_reward))
        time.sleep(1)
