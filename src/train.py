from rl.ehdqn import EHDQN
from environment_manager import create_environment
import logging
import logger as log
import numpy as np
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
                gamma_macro=args.gamma_macro,
                reward_rescale=args.reward_rescale,
                logger=log.TB_LOGGER
                )

    train_steps = 0
    tot_succ = 0
    episode_duration = 0
    for i in range(args.episodes):
        if i % 5 == 0:
            logging.info('Episode: %s' % i)

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

            obs = obs_new
            episode_duration += 1
            if is_terminal:
                log.TB_LOGGER.log_scalar(tag='Episode Duration', value=episode_duration)
                episode_duration = 0
                break

        episodes_per_epoch = 75
        if i % episodes_per_epoch == 0 and i > 0:
            log.TB_LOGGER.log_scalar(tag='Train Reward:', value=tot_succ / episodes_per_epoch)
            tot_succ = 0

        if i % 100 == 0:
            n_eval_episodes = 20
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

        # Save ckpt
        if i % 100 == 0 and i > 0:
            dqn.save(i)
