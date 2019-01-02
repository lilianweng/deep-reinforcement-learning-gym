from collections import namedtuple

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

from playground.policies.base import BaseTFModelMixin, Policy, ReplayMemory, TrainConfig
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import dense_nn


class DDPGPolicy(Policy, BaseTFModelMixin):

    def __init__(self, env, name, training=True, gamma=0.9,
                 actor_layers=[64, 32], critic_layers=[128, 64],  **kwargs):
        Policy.__init__(self, env, name, training=training, gamma=gamma, **kwargs)
        BaseTFModelMixin.__init__(self, name)

        assert isinstance(self.env.action_space, Box), \
            "Current implementation only works for continuous action space."

        self.actor_layers = actor_layers
        self.critic_layers = critic_layers

    def act(self, state, eps=0.25):
        # add random gaussian noise for action exploration.
        action = self.sess.run(self.mu, {self.s: [state]})[0]
        action += eps * np.random.randn(*self.act_dim)
        action = np.clip(action * self.env.action_space.high, self.env.action_space.low, self.env.action_space.high)
        return action

    def _build_networks(self):
        """For continuous action space.
        """
        # Define input placeholders
        self.s = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='state')
        self.a = tf.placeholder(tf.float32, shape=[None] + self.act_dim, name='action')
        self.s_next = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='next_state')
        self.r = tf.placeholder(tf.float32, shape=[None, ], name='reward')

        with tf.variable_scope('primary'):
            # Actor: deterministic policy mu(s) outputs one action vector.
            self.mu = dense_nn(self.s, self.actor_layers + self.act_dim, output_fn=tf.nn.tanh, name='mu')
            # Critic: action value, Q(s, a)
            self.Q = dense_nn(tf.concat([self.s, self.a], axis=1), self.critic_layers + [1], name='Q')
            # We want to train mu network to maximize Q value that is estimated by our critic;
            # this is only used for training.
            self.Q_mu = dense_nn(tf.concat([self.s, self.mu], axis=1), self.critic_layers + [1], name='Q', reuse=True)

        with tf.variable_scope('target'):
            # Clone target networks.
            self.mu_target = dense_nn(self.s_next, self.actor_layers + self.act_dim, output_fn=tf.nn.tanh, name='mu')
            self.Q_target = dense_nn(tf.concat([self.s_next, self.mu_target], axis=1),
                                     self.critic_layers + [1], name='Q')

        self.Q_vars = self.get_vars('primary/Q')
        self.mu_vars = self.get_vars('primary/mu')

        # sanity check
        self.primary_vars = self.Q_vars + self.mu_vars
        self.target_vars = self.get_vars('target/Q') + self.get_vars('target/mu')
        assert len(self.primary_vars) == len(self.target_vars)

    def init_target_net(self):
        self.sess.run([v_t.assign(v) for v_t, v in zip(self.target_vars, self.primary_vars)])

    def update_target_net(self, tau=0.01):
        self.sess.run([v_t.assign((1.0 - tau) * v_t + tau * v) for v_t, v in zip(self.target_vars, self.primary_vars)])

    def _build_train_ops(self):
        self.lr_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_actor')
        self.lr_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_critic')
        self.done = tf.placeholder(tf.float32, shape=None, name='terminal_flag')

        with tf.variable_scope('Q_train'):
            self.Q_reg = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.Q_vars])
            # use tf.stop_gradient() because we don't want to update the Q target net yet.
            y = self.r + self.gamma * self.Q_target * (1.0 - self.done)
            self.Q_loss = tf.reduce_mean(tf.square(tf.stop_gradient(y) - self.Q)) + 0.0001 * self.Q_reg
            # self.Q_train_op = tf.train.AdamOptimizer(self.lr_c).minimize(self.Q_loss, var_list=self.Q_vars)

            Q_optim = tf.train.AdamOptimizer(self.lr_c)
            self.Q_grads = Q_optim.compute_gradients(self.Q_loss, self.Q_vars)
            self.Q_train_op = Q_optim.apply_gradients(self.Q_grads)

        with tf.variable_scope('mu_train'):
            self.mu_loss = -tf.reduce_mean(self.Q_mu)
            self.mu_train_op = tf.train.AdamOptimizer(self.lr_a).minimize(self.mu_loss, var_list=self.mu_vars)

        with tf.variable_scope('summary'):
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')  # just for logging.
            self.summary = [
                tf.summary.scalar('loss/Q', self.Q_loss),
                tf.summary.scalar('loss/Q_reg', self.Q_reg),
                tf.summary.scalar('loss/mu', self.mu_loss),
                tf.summary.scalar('output/Q', tf.reduce_mean(self.Q)),
                tf.summary.histogram('output/Q_mu', tf.reduce_mean(self.Q_mu)),
                tf.summary.scalar('output/Q_target', tf.reduce_mean(self.Q_target)),
                tf.summary.histogram('output/mu', self.mu),
                tf.summary.histogram('output/mu_target', self.mu_target),
                tf.summary.scalar('output/episode_reward', self.ep_reward)
            ] + [
                tf.summary.scalar('grads/Q_' + var.name, tf.norm(grad))
                for grad, var in self.Q_grads if grad is not None
            ]

            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        self.train_ops = [self.Q_train_op, self.mu_train_op]

        self.sess.run(tf.global_variables_initializer())
        self.init_target_net()

    def build(self):
        self._build_networks()
        self._build_train_ops()

    class TrainConfig(TrainConfig):
        lr_a = 0.0001
        lr_c = 0.001
        # action exploration noise
        epsilon = 0.25
        epsilon_final = 0.0
        # for target network polyak averaging
        tau = 0.001

    def train(self, config: TrainConfig):
        # Construct the replay memory buffer.
        BufferRecord = namedtuple('Record', ['s', 'a', 'r', 's_next', 'done'])
        buffer = ReplayMemory(tuple_class=BufferRecord)

        step = 0
        n_episode = 0

        episode_reward = 0.
        episode_step = 0
        reward_history = []
        reward_averaged = []
        episode_length = []

        eps = config.epsilon
        eps_drop_per_step = (eps - config.epsilon_final) / config.warmup_steps
        print("decrease `epsilon` per step:", eps_drop_per_step)

        env = self.env
        ob = env.reset()
        done = False

        while step < config.n_steps:
            while not done:
                a = self.act(ob, eps)
                ob_next, r, done, _ = env.step(a)
                if done and config.done_rewards:
                    r += config.done_rewards
                step += 1
                episode_step += 1
                episode_reward += r

                buffer.add(BufferRecord(ob, a, r, ob_next, float(done)))
                ob = ob_next

                if eps > config.epsilon_final:
                    eps = max(config.epsilon_final, eps - eps_drop_per_step)

                if reward_history and config.log_every_step and step % config.log_every_step == 0:
                    # Report the performance every `log_every_step` steps
                    print("[episodes:{}/step:{}], best(reward):{:.2f}, avg(reward):{:.2f}, avg(episode length):{:.2f}, "
                          "eps:{:.4f}".format(n_episode, step, np.max(reward_history), np.mean(reward_history[-10:]),
                                              np.mean(episode_length[-10:]), eps))
                    # self.save_model(step=step)

                if buffer.size >= config.batch_size:
                    batch = buffer.pop(config.batch_size)
                    _, q_loss, mu_loss, summ_str = self.sess.run(
                        [self.train_ops, self.Q_loss, self.mu_loss, self.merged_summary], feed_dict={
                            self.lr_a: config.lr_a,
                            self.lr_c: config.lr_c,
                            self.done: batch['done'],
                            self.s: batch['s'],
                            self.a: batch['a'],
                            self.r: batch['r'],
                            self.s_next: batch['s_next'],
                            self.ep_reward: reward_history[-1] if reward_history else 0.0,
                        })
                    self.update_target_net(tau=config.tau)
                    self.writer.add_summary(summ_str, step)

            # one trajectory is complete.
            n_episode += 1
            ob = env.reset()
            done = False
            episode_length.append(episode_step)
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            episode_step = 0
            episode_reward = 0.

        self.save_model(step=step)

        print("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
            'episode_length': episode_length,
        }
        plot_learning_curve(self.model_name, data_dict, xlabel='episode')
