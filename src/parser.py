import argparse
args = argparse.ArgumentParser()


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


args.add_argument('--env', default='DimGrid-v0', type=str, help='Environment name')
args.add_argument('--ckpt', default='./', type=str, help='Ckpt path')
args.add_argument('--step', default=-1, type=int, help='Ckpt step value')
args.add_argument('--n_proc', default=1, type=int, help='Number of parallel (synced) environments')
args.add_argument('--episodes', default=1_000_000, type=int, help='number of episodes')
args.add_argument('--gamma', default=0.9, type=float, help='Discount reward factor')
args.add_argument('--gamma_macro', default=0.9, type=float, help='Discount reward factor for macro policy')
args.add_argument('--hidd_ch', default=128, type=int, help='Number of hidden units per hidden channels')
args.add_argument('--lam', default=0.1, type=float, help='Scaler for intrinsic reward')
args.add_argument('--embed_state_size', default=128, type=int, help='Number of units for embed representation')
args.add_argument('--max_time', default=4, type=int, help='Number of steps per policy')
args.add_argument('--n_subpolicy', default=2, type=int, help='Number of sub policies')
args.add_argument('--lr', default=1e-3, type=float, help='Learning rate for agent training')
args.add_argument('--eps', default=0.5, type=float, help='Chance of taking random action')
args.add_argument('--eps_decay', default=2e-5, type=float, help='Decay for macro eps')
args.add_argument('--eps_sub', default=0.5, type=float, help='Chance of taking random action for sub policy')
args.add_argument('--eps_sub_decay', default=2e-5, type=float, help='Decay for sub policy eps')
args.add_argument('--bs', default=32, type=int, help='Batch size')
args.add_argument('--train_interval', default=5, type=int, help='Steps of env before training')
args.add_argument('--train_steps', default=1, type=int, help='Steps of training')
args.add_argument('--target_int', default=100, type=int, help='Steps of training to update target network in dueling arch.')
args.add_argument('--max_memory', default=100_000, type=int, help='Max macro memory')
args.add_argument('--max_memory_sub', default=100_000, type=int, help='Max memory sub')
args.add_argument('--size', default=20, type=int, help='Size for DimGrid environment')
args.add_argument('--tau', default=0.01, type=float, help='Weight for agent loss')
args.add_argument('--beta', default=0.2, type=float, help='Weight for fwd vs inv icm loss')
args.add_argument('--reward_rescale', default=0., type=float, help='Reward rescaling: 0 -> Id, 1 -> sign(R), 2 -> [-1;1], float -> Rew * mult')


