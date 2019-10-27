from rl.ehdqn import EHDQN
from environment_manager import create_environment
import logging
import numpy as np
import parser
from tensorboard_logging import Logger
import settings as sett
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


if __name__ == '__main__':
    args = parser.args.parse_args()

    # Setup env
    env = create_environment(args.env, n_env=args.n_proc, size=args.size)

    # Logger
    TB_LOGGER = Logger(sett.LOGPATH)
    print('Torch Device: %s' % sett.device)

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
                logger=TB_LOGGER
                )

    train_steps = 0
    tot_succ = 0
    episodes_per_epoch = 50
    episode_duration = np.zeros((args.n_proc,), dtype=np.int)
    remotes = env._get_target_remotes(range(args.n_proc))
    total_episodes = 0
    i = 0
    for _ in range(args.episodes // episodes_per_epoch):

        TB_LOGGER.step += 1
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
                    TB_LOGGER.log_scalar(tag='Episode Duration', value=episode_duration[j])
                    episode_duration[j] = 0
                    remotes[j].send(('reset', None))
                    obs[j] = remotes[j].recv()
                    i += 1
                    total_episodes += 1
                    if total_episodes % 5 == 0:
                        logging.info('Simulated %s episodes' % total_episodes)

            if i > episodes_per_epoch:
                break

        TB_LOGGER.log_scalar(tag='Train Reward:', value=tot_succ / i)
        tot_succ = 0

        dqn.set_mode(training=False)
        n_eval_episodes = 25
        tot_reward = np.zeros((args.n_proc,), dtype=np.int)
        cumulative_reward = 0
        counter = 0
        obs = env.reset()
        while counter < n_eval_episodes:
            action = dqn.act(obs, deterministic=True)
            obs_new, r, is_terminal, _ = env.step(action)

            #env.render()
            tot_reward += r
            obs = obs_new

            for j, terminal in enumerate(is_terminal):
                if terminal:
                    remotes[j].send(('reset', None))
                    obs[j] = remotes[j].recv()
                    cumulative_reward += tot_reward[j]
                    tot_reward[j] = 0
                    counter += 1

        eval_succ = cumulative_reward / counter
        logging.info('Mean Reward: %s' % (eval_succ))
        TB_LOGGER.log_scalar(tag='Eval Reward', value=eval_succ)
        dqn.set_mode(training=True)

        # Save ckpt
        dqn.save(total_episodes)
