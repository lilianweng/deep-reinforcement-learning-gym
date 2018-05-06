from collections import deque

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

from playground.policies.base import BaseTFModelMixin, Policy
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import mlp_net, conv2d_net


class ReplayMemory(object):
    def __init__(self, capacity=None, replace=False):
        self.buffer = deque(maxlen=capacity)
        self.replace = replace

    def add(self, s, a, r, s_next, done):
        """(s, a, r, s_next, done)
        done (bool): whether the new state `s_next` is finished.
        """
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        assert len(self.buffer) >= batch_size

        idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=self.replace)
        selected = [self.buffer[i] for i in idxs]

        return {
            'state': [tup[0] for tup in selected],
            'action': [tup[1] for tup in selected],
            'reward': [tup[2] for tup in selected],
            'state_next': [tup[3] for tup in selected],
            'done': [tup[4] for tup in selected],
        }

    @property
    def size(self):
        return len(self.buffer)


class DqnPolicy(Policy, BaseTFModelMixin):
    def __init__(self, env, name,
                 training=True,
                 gamma=0.99,
                 lr=0.001,
                 lr_decay=1.0,
                 epsilon=1.0,
                 epsilon_final=0.01,
                 batch_size=64,
                 memory_capacity=None,
                 q_model_type='mlp',
                 q_model_params=None,
                 target_update_type='hard',
                 target_update_params=None,
                 double_q=False,
                 dueling=False):
        """
        Q func: cnn

        DQN:
            target = reward(s,a) + gamma * max(Q(s')
        """
        Policy.__init__(self, env, name, gamma=gamma, training=training)
        BaseTFModelMixin.__init__(self, name, saver_max_to_keep=5)

        assert isinstance(self.env.action_space, Discrete)
        assert isinstance(self.env.observation_space, Box)
        assert q_model_type in ('mlp', 'cnn', 'rnn')
        assert target_update_type in ('hard', 'soft')

        self.gamma = gamma
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.training = training

        self.q_model_type = q_model_type
        self.q_model_params = q_model_params or {}
        self.double_q = double_q
        self.dueling = dueling
        self.target_update_type = target_update_type
        self.target_update_every_step = (target_update_params or {}).get('every_step', 100)
        self.target_update_tau = (target_update_params or {}).get('tau', 0.05)

        self.memory = ReplayMemory(capacity=memory_capacity)
        self.batch_size = batch_size

    @property
    def act_size(self):
        # Returns: An int
        return self.env.action_space.n

    @property
    def obs_size(self):
        # Returns: A list
        sample = self.env.observation_space.sample()
        if self.q_model_type == 'mlp':
            return [sample.flatten().shape[0]]
        elif self.q_model_type == 'cnn':
            return list(sample.shape)
        else:
            assert NotImplementedError()

    def obs_to_inputs(self, ob):
        if self.q_model_type == 'mlp':
            return ob.flatten()
        elif self.q_model_type == 'cnn':
            return ob
        else:
            assert NotImplementedError()

    def _scope_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(res) > 0
        return res

    def _init_target_q_net(self):
        self.sess.run([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def _update_target_q_net_hard(self):
        self.sess.run([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def _update_target_q_net_soft(self, tau=0.05):
        self.sess.run([v_t.assign(v_t * (1. - tau) + v * tau)
                       for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def build(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')

        self.states = tf.placeholder(tf.float32, shape=[None] + self.obs_size, name='state')
        self.states_next = tf.placeholder(tf.float32, shape=[None] + self.obs_size, name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.rewards = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.done_flags = tf.placeholder(tf.float32, shape=(None,), name='done')  # binary

        if self.q_model_type == 'mlp':
            # The output is a probability distribution over all the actions.
            # layers_sizes = [256, 128, 64] + [self.act_size]
            layers_sizes = self.q_model_params.get('layer_sizes', [256, 128, 64])
            self.q = mlp_net(self.states, layers_sizes + [self.act_size], name='Q_main')
            self.q_target = mlp_net(self.states_next, layers_sizes + [self.act_size], name='Q_target')

        elif self.q_model_type == 'cnn':
            self.q = conv2d_net(self.states, self.act_size, name='Q_main')
            self.q_target = conv2d_net(self.states_next, self.act_size, name='Q_target')

        else:
            assert NotImplementedError()

        self.q_vars = self._scope_vars('Q_main')
        self.q_target_vars = self._scope_vars('Q_target')
        assert len(self.q_vars) == len(self.q_target_vars)

        print([v.name for v in self.q_vars])
        print([v.name for v in self.q_target_vars])

        max_q_next_target = tf.reduce_max(self.q_target, axis=1)
        y = (1. - self.done_flags) * self.gamma * max_q_next_target + self.rewards

        action_one_hot = tf.one_hot(self.actions, self.act_size, 1.0, 0.0, name='action_one_hot')
        pred = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

        self.loss = tf.reduce_mean(tf.square(pred - y), name="loss_mse_train")
        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss, name="adam_optim")

        with tf.variable_scope('summary'):
            q_summ = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.act_size):
                q_summ.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summ = tf.summary.merge(q_summ, 'q_summary')

            self.q_y_summ = tf.summary.histogram("batch/y", y)
            self.q_pred_summ = tf.summary.histogram("batch/pred", pred)
            self.loss_summ = tf.summary.scalar("loss", self.loss)

            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.ep_reward_summ = tf.summary.scalar('episode_reward', self.ep_reward)
            self.merged_summary = tf.summary.merge([
                self.q_y_summ, self.q_pred_summ, self.q_summ,
                self.loss_summ, self.ep_reward_summ])

        self.sess.run(tf.global_variables_initializer())
        self._init_target_q_net()

    def update_target_q_net(self, step):
        if self.target_update_type == 'hard':
            if step % self.target_update_every_step == 0:
                self._update_target_q_net_hard()
        else:
            self._update_target_q_net_soft(self.target_update_tau)

    def act(self, state, epsilon=0.0):
        if self.training and np.random.random() < epsilon:
            return self.env.action_space.sample()

        with self.sess.as_default():
            predicted_proba = self.q.eval({self.states: [state]})[0]

        best_action = max(range(self.act_size), key=lambda idx: predicted_proba[idx])
        return best_action

    def train(self, n_episodes, annealing_episodes=None, done_reward=None, every_episode=None):
        reward = 0.
        reward_history = [0.0]
        reward_averaged = []

        lr = self.lr
        eps = self.epsilon
        annealing_episodes = annealing_episodes or n_episodes
        eps_drop = (self.epsilon - self.epsilon_final) / annealing_episodes
        print("eps_drop:", eps_drop)
        step = 0

        for n_episode in range(n_episodes):
            ob = self.env.reset()
            done = False

            while not done:
                a = self.act(ob, eps)
                new_ob, r, done, info = self.env.step(a)
                step += 1
                reward += r

                # Special reward or penalty when done.
                if done and done_reward:
                    r = done_reward
                self.memory.add(self.obs_to_inputs(ob), a, r, self.obs_to_inputs(new_ob), done)

                ob = new_ob

                # No enough samples in the buffer yet.
                if self.memory.size < self.batch_size:
                    continue

                # Training with a mini batch of samples!
                batch_data_dict = self.memory.sample(self.batch_size)
                _, q_val, q_target_val, loss, summ_str = self.sess.run(
                    [self.optimizer, self.q, self.q_target, self.loss, self.merged_summary], {
                        self.learning_rate: lr,
                        self.states: batch_data_dict['state'],
                        self.actions: batch_data_dict['action'],
                        self.rewards: batch_data_dict['reward'],
                        self.states_next: batch_data_dict['state_next'],
                        self.done_flags: batch_data_dict['done'],
                        self.ep_reward: reward_history[-1],
                    })
                self.writer.add_summary(summ_str, step)
                self.update_target_q_net(step)

            # One episode is complete.
            reward_history.append(reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            reward = 0.

            # Annealing the learning and exploration rate after every episode.
            lr *= self.lr_decay
            if eps > self.epsilon_final:
                eps -= eps_drop

            if reward_history and every_episode and n_episode % every_episode == 0:
                # Report the performance every `every_step` steps
                print("[episodes:{}/step:{}], best:{}, avg:{:.2f}:{}, lr:{:.4f}, eps:{:.4f}".format(
                    n_episode, step, np.max(reward_history),
                    np.mean(reward_history[-10:]), reward_history[-5:],
                    lr, eps, self.memory.size
                ))
                # self.save_model(step=step)

        self.save_model(step=step)

        print("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
        }
        plot_learning_curve(self.model_name, data_dict, xlabel='episode')

    def test(self, n_episodes):
        reward_history = []
        reward = 0.

        for i in range(n_episodes):
            ob = self.env.reset()
            done = False
            while not done:
                a = self.act(ob)
                new_ob, r, done, _ = self.env.step(a)
                self.env.render()
                reward += r
                ob = new_ob

            reward_history.append(reward)
            reward = 0.

        print("Avg. reward over {} episodes: {:.4f}".format(n_episodes, np.mean(reward_history)))
