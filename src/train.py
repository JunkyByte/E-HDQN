from rl.ehdqn import EHDQN
from rl.dqn import DQN
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
    env = create_environment(args.env, n_env=args.n_proc, size=args.size, sparse=args.sparse)
    eval_env = create_environment(args.env, n_env=args.n_proc, seed=42, size=args.size, sparse=args.sparse)
    is_mario = True if 'Mario' in args.env else False
    norm_input = True

    # Logger
    TB_LOGGER = Logger(sett.LOGPATH)
    print('Torch Device: %s' % sett.device)

    # Store HYPER in the log
    for key, value in args._get_kwargs():
        TB_LOGGER.log_text(tag=str(key), value=[str(value)], step=0)

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
                  n_proc=args.n_proc,
                  reward_rescale=args.reward_rescale,
                  logger=TB_LOGGER,
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
                    n_proc=args.n_proc,
                    gamma_macro=args.gamma_macro,
                    reward_rescale=args.reward_rescale,
                    logger=TB_LOGGER,
                    norm_input=norm_input
                    )

    # Load ckpt to fine-tune if specified
    if args.ckpt != parser.args.get_default('ckpt'):
        print(args.ckpt, parser.args.get_default('ckpt'))
        logging.info('Fine tuning mode has been specified, loading ckpt %s' % args.ckpt)
        dqn.eps_sub = 0.25
        if hasattr(dqn, 'eps'):
            dqn.eps = 0
        dqn.load(args.ckpt, i=args.step)

    train_steps = 0
    tot_succ = 0
    episodes_per_epoch = args.episodes_per_epoch
    episode_duration = np.zeros((args.n_proc,), dtype=np.float)
    total_episodes = 0
    total_x_pos = 0
    i = 0
    for _ in range(args.episodes // episodes_per_epoch):

        TB_LOGGER.step += 1
        obs = env.reset()

        i = 0
        n_info = 0
        while True:
            action = dqn.act(obs)
            obs_new, r, is_terminal, info = env.step(action)

            tot_succ += sum(r)

            obs_store = [s if not is_terminal[k] else info[k]['terminal_observation'] for k, s in enumerate(obs_new)]
            dqn.store_transition(obs, obs_store, action, r, is_terminal)
            #env.render()

            train_steps += 1
            if train_steps % max(1, args.train_interval // args.n_proc) == 0 and train_steps > 0:
                train_steps = 0
                dqn.update()

            obs = obs_new
            episode_duration += 1
            for j, terminal in enumerate(is_terminal):
                if terminal:
                    if is_mario:
                        try:
                            total_x_pos += info[i]['x_pos']
                            n_info += 1
                        except IndexError:
                            pass
                    TB_LOGGER.log_scalar(tag='Episode Duration', value=episode_duration[j])
                    episode_duration[j] = 0
                    i += 1
                    total_episodes += 1
                    if total_episodes % 5 == 0:
                        logging.info('Simulated %s episodes' % total_episodes)

            if i > episodes_per_epoch:
                break

        if is_mario:
            TB_LOGGER.log_scalar(tag='Mean End X', value=total_x_pos / n_info)
        TB_LOGGER.log_scalar(tag='Train Reward:', value=tot_succ / i)
        tot_succ = 0
        total_x_pos = 0

        dqn.set_mode(training=False)
        n_eval_episodes = 100
        tot_reward = np.zeros((args.n_proc,), dtype=np.float)
        cumulative_reward = 0
        counter = 0
        obs = eval_env.reset()
        while counter < n_eval_episodes:
            action = dqn.act(obs, deterministic=True)
            obs_new, r, is_terminal, info = eval_env.step(action)

            tot_reward += r
            obs = obs_new

            for j, terminal in enumerate(is_terminal):
                if terminal:
                    cumulative_reward += tot_reward[j]
                    tot_reward[j] = 0
                    counter += 1

        eval_succ = cumulative_reward / counter
        logging.info('Mean Reward: %s' % (eval_succ))
        TB_LOGGER.log_scalar(tag='Eval Reward', value=eval_succ)
        dqn.set_mode(training=True)

        # Save ckpts
        dqn.save(total_episodes)
