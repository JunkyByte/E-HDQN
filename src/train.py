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
    env = create_environment(args.env, n_env=args.n_proc, size=args.size)
    eval_env = create_environment(args.env, n_env=1, size=args.size, eval=True)

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
                n_proc=args.n_proc,
                gamma_macro=args.gamma_macro,
                reward_rescale=args.reward_rescale,
                logger=log.TB_LOGGER
                )

    train_steps = 0
    tot_succ = 0
    episodes_per_epoch = 100
    episode_duration = np.zeros((args.n_proc,), dtype=np.int)
    remotes = env._get_target_remotes(range(args.n_proc))
    total_episodes = 0
    i = 0
    for _ in range(args.episodes):
        logging.info('Episode: %s' % i)

        log.TB_LOGGER.step += 1
        obs = env.reset()

        i = 0
        while True:
            action = dqn.act(obs)
            obs_new, r, is_terminal, _ = env.step(action)

            tot_succ += sum(r)
            dqn.store_transition(obs, obs_new, action, r, is_terminal)

            train_steps += 1
            if train_steps % max(1, args.train_interval // args.n_proc) == 0 and train_steps > 0:
                train_steps = 0
                dqn.update()

            obs = obs_new
            episode_duration += 1
            for j, terminal in enumerate(is_terminal):
                if terminal:
                    log.TB_LOGGER.log_scalar(tag='Episode Duration', value=episode_duration[j])
                    episode_duration[j] = 0
                    remotes[j].send(('reset', None))
                    obs[j] = remotes[j].recv()
                    i += 1
                    total_episodes += 1

            if i > episodes_per_epoch:
                break

        log.TB_LOGGER.log_scalar(tag='Train Reward:', value=tot_succ / i)
        tot_succ = 0

        dqn.set_mode(training=False)
        n_eval_episodes = 25
        tot_reward = 0
        for _ in range(n_eval_episodes):
            obs = eval_env.reset()
            while True:
                action = dqn.act(obs[np.newaxis], deterministic=True)
                obs_new, r, is_terminal, _ = eval_env.step(action)

                #env.render()
                tot_reward += r
                obs = obs_new

                if is_terminal:
                    break

        eval_succ = tot_reward / n_eval_episodes
        logging.info('Mean Reward: %s' % (eval_succ))
        log.TB_LOGGER.log_scalar(tag='Eval Reward', value=eval_succ)
        dqn.set_mode(training=True)

        # Save ckpt
        dqn.save(total_episodes)
