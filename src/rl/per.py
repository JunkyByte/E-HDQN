import random
import numpy as np
from rl.SumTree import SumTree

class PERMemory:  # stored as ( s, s_, a, r, end ) in SumTree
    e = 1e-8
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 8e-6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __len__(self):
        return self.tree.n_entries

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def store_transition(self, s, s1, a, r, is_terminal):
        sample = (s, s1, a, r, is_terminal)
        p = self._get_priority(1)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1. - self.e, self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return (*list(zip(*batch)), idxs, is_weight)

    def update(self, idx, error):
        error = min(1, max(-1, error))
        p = self._get_priority(error)
        self.tree.update(idx, p)
