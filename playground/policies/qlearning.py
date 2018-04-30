from collections import defaultdict, deque

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

from playground.policies.base import BaseTFModelMixin, Policy
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import mlp


class QlearningPolicy(Policy):
    def __init__(self, env, name,
                 training=True,
                 gamma=0.99,
                 alpha=0.5,
                 alpha_decay=1.0,
                 epsilon=0.05,
                 epsilon_decay=1.0,
                 Q=None):
        """
        1. We start from state s and

        2.  At state s, with action a, we observe a reward r(s, a) and get into the
        next state s'. Update Q function:

            Q(s, a) += learning_rate * (r(s, a) + gamma * max Q(s', .) - Q(s, a))

        Repeat this process.
        """
        super().__init__(env, name, gamma=gamma, training=training)
        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Discrete)

        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.Q = Q
        self.actions = range(self.env.action_space.n)

    def build(self):
        self.Q = defaultdict(float)

    def act(self, state, epsilon):
        """Normally pick action according to Q values ~ argmax_a Q(s, a).
        Exploration is forced by epsilon-greedy.
        """
        if self.training and np.random.rand() < epsilon:
            # Let's explore!
            return self.env.action_space.sample()

        # Pick the action with highest Q value.
        qvals = {a: self.Q[state, a] for a in self.actions}
        max_q = max(qvals.values())
        actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
        return np.random.choice(actions_with_max_q)

    def _update_q_value(self, s, a, r, s_next):
        """
        Q(s, a) += alpha * (r(s, a) + gamma * max Q(s', .) - Q(s, a))
        """
        max_q_next = max([self.Q[s_next, a] for a in self.actions])
        self.Q[s, a] += self.alpha * (r + self.gamma * max_q_next - self.Q[s, a])

    def train(self, n_steps, done_reward=None, every_step=None, with_monitor=False):
        alpha_history = []
        eps_history = []
        reward_history = []
        reward_averaged = []
        reward = 0

        ob = self.env.reset()
        alpha = self.alpha
        eps = self.epsilon

        for step in range(n_steps):
            a = self.act(ob, eps)
            new_ob, r, done, info = self.env.step(a)
            reward += r

            if done:
                if done_reward is None:
                    done_reward = r

                self._update_q_value(ob, a, done_reward, new_ob)
                ob = self.env.reset()

                reward_history.append(reward)
                reward = 0
                alpha *= self.alpha_decay
                eps *= self.epsilon_decay
                # print(step, '-', self.alpha, self.epsilon)

            else:
                self._update_q_value(ob, a, r, new_ob)
                ob = new_ob

            reward_averaged.append(np.mean(reward_history[-20:]) if reward_history else None)
            alpha_history.append(alpha)
            eps_history.append(eps)

            if len(reward_history) > 0 and every_step is not None and step % every_step == 0:
                # Report the performance every 100 steps
                print("[step:{}] episodes: {}, best: {}, avg: {}, Q.size: {}".format(
                    step, len(reward_history), np.max(reward_history),
                    np.mean(reward_history[-20:]), len(self.Q)))

        print("[FINAL] Num. episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        data_dict = {
            'alpha': alpha_history,
            'epsilon': eps_history,
            'reward_averaged': reward_averaged,
        }
        plot_learning_curve(n_steps, self.name, data_dict)
