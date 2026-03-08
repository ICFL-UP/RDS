import numpy as np


class RLDatasetEnv:
    """A tiny Gym-like environment over a fixed dataset.

    - state: a single feature vector
    - action: integer label (0 or 1)
    - reward: +1 for correct prediction, -1 for incorrect
    Episode runs through the dataset once.
    """
    def __init__(self, features, labels):
        self.X = self._to_numpy(features)
        self.y = self._to_numpy(labels).astype(int)
        assert len(self.X) == len(self.y)
        self.n = len(self.y)
        self.idx = 0

    def _to_numpy(self, arr):
        try:
            return arr.toarray()
        except Exception:
            return np.array(arr)

    def reset(self):
        self.idx = 0
        if self.n == 0:
            return None
        return self._get_state()

    def _get_state(self):
        return self.X[self.idx]

    def step(self, action):
        reward = 1 if int(action) == int(self.y[self.idx]) else -1
        self.idx += 1
        done = self.idx >= self.n
        next_state = None if done else self._get_state()
        return next_state, reward, done, {}
