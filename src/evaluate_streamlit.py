import time
from rl.ehdqn import EHDQN
from rl.dqn import DQN
import logging
import argparse
import numpy as np
from environment_manager import create_environment
import parser
import pandas as pd
import streamlit as st
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
st.title('Environment Evaluation')


if __name__ == '__main__':
    args = parser.args.parse_args()
    image = st.empty()
    st.text('Subpolicy:')
    sub_policy = st.empty()
    st.text('Action:')
    act = st.empty()

    nskip = 6
    is_mario = True if 'Mario' in args.env else False
    norm_input = True
    env = args.env
    if is_mario:
        if not 'Random' in args.env:
            world = st.selectbox('World', list(range(1, 9)))
            env = 'SuperMarioBros-%s-1-v0' % world
        nskip = st.selectbox('NSkip', [6] + list(range(1, 12)))

    # Setup env
    env = create_environment(env, n_env=1, seed=42, size=args.size, nskip=nskip, sparse=args.sparse)

    obs = env.reset()

    # Setup Model
    n_actions = env.action_space.n if env.action_space.shape == () else env.action_space.shape[0]
    n_state = env.observation_space.n if env.observation_space.shape == () else env.observation_space.shape

    conv = True if isinstance(n_state, tuple) else False
    if args.use_baseline:
        dqn = DQN(state_dim=n_state,
                  tau=args.tau,
                  action_dim=n_actions,
                  gamma=args.gamma,
                  hidd_ch=args.hidd_ch,
                  lam=args.lam,
                  lr=args.lr,
                  eps_sub=args.eps_sub,
                  eps_sub_decay=args.eps_sub_decay,
                  beta=args.beta,
                  bs=args.bs,
                  target_interval=args.target_int,
                  train_steps=args.train_steps,
                  max_memory=args.max_memory,
                  conv=conv,
                  per=args.per,
                  n_proc=1,
                  reward_rescale=args.reward_rescale,
                  norm_input=norm_input
                  )
    else:
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
                    target_interval=args.target_int,
                    train_steps=args.train_steps,
                    max_memory=args.max_memory,
                    max_memory_sub=args.max_memory_sub,
                    conv=conv,
                    per=args.per,
                    n_proc=1,
                    gamma_macro=args.gamma_macro,
                    reward_rescale=args.reward_rescale,
                    norm_input=norm_input
                    )

    # Load model
    dqn.load(args.ckpt, i=args.step)
    dqn.set_mode(training=False)
    n_subpolicy = dqn.n_subpolicy if hasattr(dqn, 'n_subpolicy') else 1
    n_actions = dqn.action_dim
    size = 18

    while True:
        tot_reward = 0
        obs = env.reset()
        for i in range(0, np.random.randint(30)):
            obs, _, _, _ = env.step([env.action_space.sample()])

        while True:
            action = dqn.act(obs, deterministic=True)
            if isinstance(action, int):
                action = [action]

            if hasattr(dqn, 'selected_policy'):
                logging.info('Policy: %s Action %s' % (dqn.selected_policy, action))
            else:
                logging.info('Action: %s' % action)

            obs_new, r, is_terminal, _ = env.step(action)

            img_env = env.render(mode='rgb_array')
            image.image(img_env)

            # Display subpolicy
            img_sub = np.zeros((size, 2 * size * n_subpolicy, 3), dtype=np.uint8)
            s = dqn.selected_policy[0] if hasattr(dqn, 'selected_policy') else 0
            img_sub[:, 2 * s * size: 2 * (s + 1) * size, 0] = 255
            sub_policy.image(img_sub)

            # Display action
            img_act = np.zeros((size, 2 * size * n_actions, 3), dtype=np.uint8)
            a = action[0]
            img_act[:, 2 * a * size: 2 * (a + 1) * size, 0] = 255
            act.image(img_act)

            tot_reward += r[0]
            obs = obs_new

            if is_terminal:
                break

        logging.info('Reward: %s' % (tot_reward))
        time.sleep(1)
