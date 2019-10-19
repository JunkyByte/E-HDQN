import numpy as np

class Memory:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.state = []
        self.new_state = []
        self.action = []
        self.reward = []
        self.is_terminal = []

    def store_transition(self, s, s1, a, r, is_terminal):
        self.state.append(s)
        self.new_state.append(s1)
        self.action.append(a)
        self.reward.append(r)
        self.is_terminal.append(is_terminal)
        assert len(self.state) == len(self.new_state) == len(self.reward) == len(self.is_terminal) == len(self.action)

        if len(self.state) > self.max_memory:
            self.state.pop(0)
            self.new_state.pop(0)
            self.action.pop(0)
            self.reward.pop(0)
            self.is_terminal.pop(0)

    def clear_memory(self):
        del self.state[:]
        del self.new_state[:]
        del self.action[:]
        del self.reward[:]
        del self.is_terminal[:]

    def sample(self, bs):
        idx = np.random.randint(len(self.state), size=bs)
        state = np.array(self.state)[idx]
        new_state = np.array(self.new_state)[idx]
        action = np.array(self.action)[idx]
        reward = np.array(self.reward)[idx]
        is_terminal = 1 - np.array(self.is_terminal, dtype=np.int)[idx]
        return state, new_state, action, reward, is_terminal
