import numpy as np

class MultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.probs = np.random.rand(n_arms)

    def pull(self, arm):
        return np.random.rand() < self.probs[arm]

    def best_arm(self):
        return np.argmax(self.probs)
