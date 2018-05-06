import gym
import numpy as np

from gym.spaces import Box, Discrete


class DigitizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.ob_shape = self.observation_space.shape

        print("New ob space:", Discrete((n_bins + 1) ** low.flatten().shape[0]))
        self.observation_space = Discrete(n_bins ** low.flatten().shape[0])

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)


class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env=None, n_bins=10):
        super().__init__(env)
        assert isinstance(env.action_space, Box)
        self._dist_to_cont = []

        for low, high in zip(env.action_space.low, env.action_space.high):
            self._dist_to_cont.append(np.linspace(low, high, n_bins + 1))
        print(self._dist_to_cont)
        temp = [n_bins + 1 for _ in self._dist_to_cont]
        self.action_space = gym.spaces.MultiDiscrete(temp)

    def action(self, action):
        assert len(action) == len(self._dist_to_cont)
        return np.array([m[a] for a, m in zip(action, self._dist_to_cont)], dtype=np.float32)
