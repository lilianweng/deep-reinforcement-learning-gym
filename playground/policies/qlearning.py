from collections import defaultdict

import numpy as np
from gym.spaces import Discrete

from playground.policies.base import Policy, TrainConfig
from playground.policies.memory import Transition
from playground.utils.misc import plot_learning_curve


class QlearningPolicy(Policy):
    def __init__(self, env, name, training=True, gamma=0.99, Q=None):
        """
        This Q-learning implementation only works on an environment with discrete
        action and observation space. We use a dict to memorize the Q-value.

        1. We start from state s and

        2.  At state s, with action a, we observe a reward r(s, a) and get into the
        next state s'. Update Q function:

            Q(s, a) += learning_rate * (r(s, a) + gamma * max Q(s', .) - Q(s, a))

        Repeat this process.
        """
        super().__init__(env, name, gamma=gamma, training=training)
        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Discrete)

        self.Q = Q
        self.actions = range(self.env.action_space.n)

    def build(self):
        self.Q = defaultdict(float)

    def act(self, state, eps=0.1):
        """Pick best action according to Q values ~ argmax_a Q(s, a).
        Exploration is forced by epsilon-greedy.
        """
        if self.training and eps > 0. and np.random.rand() < eps:
            return self.env.action_space.sample()

        # Pick the action with highest Q value.
        qvals = {a: self.Q[state, a] for a in self.actions}
        max_q = max(qvals.values())

        # In case multiple actions have the same maximum Q value.
        actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
        return np.random.choice(actions_with_max_q)

    def _update_q_value(self, tr, alpha):
        """
        Q(s, a) += alpha * (r(s, a) + gamma * max Q(s', .) - Q(s, a))
        """
        max_q_next = max([self.Q[tr.s_next, a] for a in self.actions])
        # We do not include the value of the next state if terminated.
        self.Q[tr.s, tr.a] += alpha * (
            tr.r + self.gamma * max_q_next * (1.0 - tr.done) - self.Q[tr.s, tr.a]
        )

    class TrainConfig(TrainConfig):
        alpha = 0.5
        alpha_decay = 0.998
        epsilon = 1.0
        epsilon_final = 0.05
        n_episodes = 1000
        warmup_episodes = 800
        log_every_episode = 10

    def train(self, config: TrainConfig):
        reward_history = []
        reward_averaged = []
        step = 0
        alpha = config.alpha
        eps = config.epsilon

        warmup_episodes = config.warmup_episodes or config.n_episodes
        eps_drop = (config.epsilon - config.epsilon_final) / warmup_episodes

        for n_episode in range(config.n_episodes):
            ob = self.env.reset()
            done = False
            reward = 0.

            while not done:
                a = self.act(ob, eps)
                new_ob, r, done, info = self.env.step(a)
                if done and config.done_reward is not None:
                    r += config.done_reward

                self._update_q_value(Transition(ob, a, r, new_ob, done), alpha)

                step += 1
                reward += r
                ob = new_ob

            reward_history.append(reward)
            reward_averaged.append(np.average(reward_history[-50:]))

            alpha *= config.alpha_decay
            if eps > config.epsilon_final:
                eps = max(config.epsilon_final, eps - eps_drop)

            if config.log_every_episode is not None and n_episode % config.log_every_episode == 0:
                # Report the performance every 100 steps
                print("[episode:{}|step:{}] best:{} avg:{:.4f} alpha:{:.4f} eps:{:.4f} Qsize:{}".format(
                    n_episode, step, np.max(reward_history),
                    np.mean(reward_history[-10:]), alpha, eps, len(self.Q)))

        print("[FINAL] Num. episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        data_dict = {'reward': reward_history, 'reward_avg50': reward_averaged}
        plot_learning_curve(self.name, data_dict, xlabel='episode')
