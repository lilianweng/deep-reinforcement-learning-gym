from collections import defaultdict

import numpy as np
from gym.spaces import Discrete

from playground.policies.base import Policy, Transition
from playground.utils.misc import plot_learning_curve


class QlearningPolicy(Policy):
    def __init__(self, env, name,
                 training=True,
                 gamma=0.99,
                 alpha=0.5,
                 alpha_decay=1.0,
                 epsilon=1.0,
                 epsilon_final=0.01,
                 Q=None):
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

        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final

        self.Q = Q
        self.actions = range(self.env.action_space.n)

    def build(self):
        self.Q = defaultdict(float)

    def act(self, state, epsilon=0.1):
        """Pick best action according to Q values ~ argmax_a Q(s, a).
        Exploration is forced by epsilon-greedy.
        """
        if self.training and epsilon > 0. and np.random.rand() < epsilon:
            return self.env.action_space.sample()

        # Pick the action with highest Q value.
        qvals = {a: self.Q[state, a] for a in self.actions}
        max_q = max(qvals.values())

        # In case multiple actions have the same maximum Q value.
        actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
        return np.random.choice(actions_with_max_q)

    def _update_q_value(self, tr):
        """
        Q(s, a) += alpha * (r(s, a) + gamma * max Q(s', .) - Q(s, a))
        """
        max_q_next = max([self.Q[tr.s_next, a] for a in self.actions])
        # We do not include the value of the next state if terminated.
        self.Q[tr.s, tr.a] += self.alpha * (
            tr.r + self.gamma * max_q_next * (1.0 - tr.done) - self.Q[tr.s, tr.a]
        )

    def train(self, n_episodes=100, annealing_episodes=None, done_reward=None, every_episode=None):
        reward_history = []
        reward_averaged = []
        step = 0
        alpha = self.alpha
        eps = self.epsilon

        annealing_episodes = annealing_episodes or n_episodes
        eps_drop = (self.epsilon - self.epsilon_final) / annealing_episodes

        for n_episode in range(n_episodes):
            ob = self.env.reset()
            done = False
            reward = 0.

            while not done:
                a = self.act(ob, eps)
                new_ob, r, done, info = self.env.step(a)
                if done and done_reward is not None:
                    r = done_reward

                self._update_q_value(Transition(ob, a, r, new_ob, done))

                step += 1
                reward += r
                ob = new_ob

            reward_history.append(reward)
            reward_averaged.append(np.average(reward_history[-50:]))

            alpha *= self.alpha_decay
            if eps > self.epsilon_final:
                eps -= eps_drop

            if every_episode is not None and n_episode % every_episode == 0:
                # Report the performance every 100 steps
                print("[episode:{}|step:{}] best:{} avg:{:.4f}|{} alpha:{:.4f} eps:{:.4f} Qsize:{}".format(
                    n_episode, step, np.max(reward_history),
                    np.mean(reward_history[-10:]), reward_history[-5:],
                    alpha, eps, len(self.Q)))

        print("[FINAL] Num. episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        plot_learning_curve(self.name,
                            {'reward': reward_history, 'reward_avg50': reward_averaged},
                            xlabel='episode')
