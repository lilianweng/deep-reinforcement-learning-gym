from collections import deque

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

from playground.policies.base import BaseTFModelMixin, Policy, ReplayMemory, Transition
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import mlp, conv2d_net, lstm_net


class ReplaceEpisodeMemory(object):
    def __init__(self, capacity=100000, step_size=16):
        self.buffer = deque(maxlen=capacity)
        self.step_size = step_size

    def add(self, episode):
        """A list of (s, a, r, s_next, done) for a complete episode.
        """
        assert all(len(ep) == 5 for ep in episode)
        assert all(not ep[4] for ep in episode[:-1])
        assert episode[-1][4]
        if len(episode) >= self.step_size:
            self.buffer.append(episode)

    def sample(self, batch_size):
        ep_idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=True)
        selected = {
            'state': [],
            'action': [],
            'reward': [],
            'state_next': [],
            'done': [],
        }
        for ep_idx in ep_idxs:
            i = np.random.randint(0, len(self.buffer[ep_idx]) + 1 - self.step_size)
            ep_selected = self.buffer[ep_idx][i: i + self.step_size]
            selected['state'].append([tup[0] for tup in ep_selected])
            selected['action'].append([tup[1] for tup in ep_selected])
            selected['reward'].append([tup[2] for tup in ep_selected])
            selected['state_next'].append([tup[3] for tup in ep_selected])
            selected['done'].append([tup[4] for tup in ep_selected])

        return {k: np.array(v) for k, v in selected.items()}

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
                 memory_capacity=100000,
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
        assert q_model_type in ('mlp', 'conv', 'lstm')
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

        if self.q_model_type == 'rnn':
            self.step_size = q_model_params.get('step_size', 32)
            self.memory = ReplaceEpisodeMemory(capacity=memory_capacity, step_size=self.step_size)
        else:
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
        elif self.q_model_type == 'conv':
            return list(sample.shape)
        elif self.q_model_type == 'lstm':
            return list(sample.shape)
        else:
            assert NotImplementedError()

    def obs_to_inputs(self, ob):
        if self.q_model_type == 'mlp':
            return ob.flatten()
        elif self.q_model_type == 'conv':
            return ob
        elif self.q_model_type == 'lstm':
            return ob
        else:
            assert NotImplementedError()

    def _init_target_q_net(self):
        self.sess.run([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def _update_target_q_net_hard(self):
        self.sess.run([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def _update_target_q_net_soft(self, tau=0.05):
        self.sess.run([v_t.assign(v_t * (1. - tau) + v * tau)
                       for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def _create_mlp_net(self):
        self.states = tf.placeholder(tf.float32, shape=(None, self.obs_size), name='state')
        self.states_next = tf.placeholder(tf.float32, shape=(None, self.obs_size), name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(None, ), name='action')
        self.actions_next = tf.placeholder(tf.int32, shape=(None, ), name='action_next') # determined by the primary network
        self.rewards = tf.placeholder(tf.float32, shape=(None, ), name='reward')
        self.done_flags = tf.placeholder(tf.float32, shape=(None, ), name='done')  # binary

        # The output is a probability distribution over all the actions.
        # layers_sizes = [256, 128, 64] + [self.act_size]
        layers_sizes = self.q_model_params.get('layer_sizes', [256, 128, 64])
        self.q = mlp(self.states, layers_sizes + [self.act_size], name='Q_primary')
        self.q_target = mlp(self.states_next, layers_sizes + [self.act_size], name='Q_target')

    def _create_conv_net(self):
        pass

    def _create_lstm_net(self):
        pass

    def create_primary_and_target_q_networks(self):
        pass

    def build(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')

        if self.q_model_type == 'lstm':
            step_size_list = [self.step_size]
        else:
            step_size_list = []

        self.states = tf.placeholder(tf.float32, shape=(None, *(step_size_list + self.obs_size)), name='state')
        self.states_next = tf.placeholder(tf.float32, shape=(None, *(step_size_list + self.obs_size)), name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(None, *step_size_list), name='action')
        self.rewards = tf.placeholder(tf.float32, shape=(None, *step_size_list), name='reward')
        self.done_flags = tf.placeholder(tf.float32, shape=(None, *step_size_list), name='done')  # binary

        if self.q_model_type == 'mlp':
            # The output is a probability distribution over all the actions.
            # layers_sizes = [256, 128, 64] + [self.act_size]
            layers_sizes = self.q_model_params.get('layer_sizes', [256, 128, 64])
            self.q = mlp(self.states, layers_sizes + [self.act_size], name='Q_primary')
            self.q_target = mlp(self.states_next, layers_sizes + [self.act_size], name='Q_target')

        elif self.q_model_type == 'conv':
            self.q = conv2d_net(self.states, self.act_size, name='Q_primary')
            self.q_target = conv2d_net(self.states_next, self.act_size, name='Q_target')

        elif self.q_model_type == 'lstm':
            lstm_layers = self.q_model_params.get('lstm_layers', 1)
            lstm_size = self.q_model_params.get('lstm_size', 256)
            self.q, _ = lstm_net(self.states, self.act_size, name='Q_primary',
                                     lstm_layers=lstm_layers, lstm_size=lstm_size)
            self.q_target, _ = lstm_net(self.states_next, self.act_size, name='Q_target',
                                     lstm_layers=lstm_layers, lstm_size=lstm_size)
        else:
            assert NotImplementedError()

        self.q_vars = self.scope_vars('Q_primary')
        self.q_target_vars = self.scope_vars('Q_target')
        assert len(self.q_vars) == len(self.q_target_vars)

        self.actions_selected_by_q = tf.argmax(self.q, axis=-1, name='action_selected')
        action_one_hot = tf.one_hot(self.actions, self.act_size, 1.0, 0.0, name='action_one_hot')
        pred = tf.reduce_sum(self.q * action_one_hot, reduction_indices=-1, name='q_acted')

        if self.double_q:
            self.actions_next = tf.placeholder(tf.int32, shape=(None, *step_size_list), name='action_next')
            actions_next_flatten = tf.range(0, self.batch_size) * self.q_target.shape[1] + self.actions_next
            max_q_next_target = tf.gather(tf.reshape(self.q_target, [-1]), actions_next_flatten)
        else:
            max_q_next_target = tf.reduce_max(self.q_target, axis=-1)

        y = self.rewards + (1. - self.done_flags) * self.gamma * max_q_next_target

        self.loss = tf.reduce_mean(tf.square(pred - tf.stop_gradient(y)), name="loss_mse_train")
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

    def act(self, state, epsilon=0.):
        if self.training and np.random.random() < epsilon:
            return self.env.action_space.sample()

        with self.sess.as_default():
            if self.q_model_type == 'lstm':
                predicted_proba = self.q.eval({self.states: [
                    [np.zeros(state.shape)] * (self.step_size - 1) + [state]
                ]})[0][-1]
                return max(range(self.act_size), key=lambda idx: predicted_proba[idx])

            else:
                return self.actions_selected_by_q.eval({self.states: [state]})[0]

    def train(self, n_episodes, annealing_episodes=None, every_episode=None):
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
            # traj_records = []

            while not done:
                a = self.act(ob, eps)
                new_ob, r, done, info = self.env.step(a)
                step += 1
                reward += r

                #traj_records.append(
                #    (self.obs_to_inputs(ob), a, r, self.obs_to_inputs(new_ob), done)
                #)
                self.memory.add(Transition(ob, a, r, new_ob, done))

                ob = new_ob

                # No enough samples in the buffer yet.
                if self.memory.size < self.batch_size:
                    continue

                # Training with a mini batch of samples!
                batch_data = self.memory.sample(self.batch_size)
                feed_dict = {
                        self.learning_rate: lr,
                        self.states: batch_data['s'],
                        self.actions: batch_data['a'],
                        self.rewards: batch_data['r'],
                        self.states_next: batch_data['s_next'],
                        self.done_flags: batch_data['done'],
                        self.ep_reward: reward_history[-1],
                    }

                if self.double_q:
                    actions_next = self.sess.run(self.actions_selected_by_q, {
                        self.states: batch_data['s_next']
                    })
                    feed_dict.update({self.actions_next: actions_next})

                _, q_val, q_target_val, loss, summ_str = self.sess.run(
                    [self.optimizer, self.q, self.q_target, self.loss, self.merged_summary],
                    feed_dict
                )
                self.writer.add_summary(summ_str, step)
                self.update_target_q_net(step)

            ## Add all the (s, a, r, s', done) of one trajectory into the replay memory.
            #self.memory.add(traj_records)

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

