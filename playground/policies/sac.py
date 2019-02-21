import logging
import numpy as np
import tensorflow as tf

from gym.spaces import Discrete

from playground.policies.base import BaseModelMixin, BaseTrainConfig, Policy
from playground.policies.memory import ReplayMemory, Transition
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import dense_nn


class SACPolicy(Policy, BaseModelMixin):

    def __init__(self, env, name, training=True, gamma=0.99, layer_sizes=None, clip_norm=None, **kwargs):
        Policy.__init__(self, env, name, training=training, gamma=gamma, **kwargs)
        BaseModelMixin.__init__(self, name)

        assert isinstance(self.env.action_space, Discrete), \
            "Current SACPolicy implementation only works for discrete action space."

        self.layer_sizes = [64, 64] if layer_sizes is None else layer_sizes
        self.clip_norm = clip_norm
        self.entropy_target = -self.act_size

    def act(self, state):
        # Discrete actions
        proba = self.sess.run(tf.nn.softmax(self.pi), feed_dict={self.s: [state]})[0]
        return max(range(self.act_size), key=lambda i: proba[i])

    def _build_networks(self):
        # Define input placeholders
        self.s = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='state')
        self.a = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.s_next = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='next_state')
        self.r = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.done = tf.placeholder(tf.float32, shape=(None,), name='done_flag')

        # Network architectures.
        pi_network_arch = self.layer_sizes + [self.act_size]
        Q_network_arch = self.layer_sizes + [self.act_size]

        # Policy: predict action probabilities
        self.pi = dense_nn(self.s, pi_network_arch, name='pi')
        self.pi_next = dense_nn(self.s_next, pi_network_arch, name='pi', reuse=True)  # a_t+1 ~ pi(.|s_t+1)
        self.pi_vars = self.scope_vars('pi')

        # Q function: we would like to learn two soft Q functions independently to
        # mitigate bias introduced during the policy improvement.
        self.Q = dense_nn(self.s, Q_network_arch, name='Q')
        self.Q_next = dense_nn(self.s_next, Q_network_arch, name='Q', reuse=True)  # Q(s_t+1, a_t+1)
        self.Q_vars = self.scope_vars('Q')

        # Q function targets
        self.Q_target = dense_nn(self.s, Q_network_arch, name='Q_target')
        self.Q_target_next = dense_nn(self.s, Q_network_arch, name='Q_target', reuse=True)
        self.Q_target_vars = self.scope_vars('Q_target')
        self.update_Q_target_op = [v_t.assign(v) for v_t, v in zip(self.Q_target_vars, self.Q_vars)]

        # Temperature
        self.alpha = dense_nn(self.s, [64, 1], name='alpha')
        self.alpha_vars = self.scope_vars('alpha')

    def _build_optimization_op(self, loss, vars, lr):
        optim = tf.train.AdamOptimizer(lr)
        grads = optim.compute_gradients(loss, vars)
        if self.clip_norm:
            grads = [(tf.clip_by_norm(g, self.clip_norm), v) for g, v in grads]
        train_op = optim.apply_gradients(grads)
        return train_op

    def _build_train_ops(self):
        self.lr = tf.placeholder(tf.float32, shape=None, name='learning_rate')

        with tf.variable_scope('training_Q'):
            # Compute Q(a_t)
            q = tf.reduce_sum(tf.one_hot(self.a, self.act_size, 1.0, 0.0) * self.Q, axis=1)

            # Compute the soft V(s_{t+1}) according to Bellman equation, using target Q func.
            a_ent_next = - tf.reduce_sum(self.pi_next * tf.log(self.pi_next), axis=1)  # H(pi(a_t+1|s_t+1))
            self.soft_V_next = tf.reduce_sum(self.pi_next * self.Q_target_next, axis=1) + self.alpha * a_ent_next

            # loss func for Q is MSE
            y = self.r + self.gamma * self.soft_V_next * (1.0 - self.done)  # regression target.
            loss_Q = tf.reduce_mean(tf.square(q - y))
            train_Q_op = self._build_optimization_op(loss_Q, self.Q_vars, self.lr)

        with tf.variable_scope('training_pi'):
            a_ent = - tf.reduce_sum(self.pi * tf.log(self.pi), axis=1)  # H(pi(a_t|s_t))

            # The policy is trained to minimize KL divergence
            loss_pi = tf.reduce_mean(- self.alpha * a_ent - tf.reduce_sum(self.pi * self.Q, axis=1))
            train_pi_op = self._build_optimization_op(loss_pi, self.pi_vars, self.lr)

        with tf.variable_scope('training_alpha'):
            a_ent = - tf.reduce_sum(self.pi * tf.log(self.pi), axis=1)  # H(pi(a_t|s_t))

            loss_alpha = tf.reduce_mean(self.alpha * a_ent - self.alpha * self.entropy_target)
            train_alpha_op = self._build_optimization_op(loss_alpha, self.alpha_vars, self.lr)

        self.train_ops = [train_Q_op, train_pi_op, train_alpha_op]

        with tf.variable_scope('summary'):
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.summary = [
                tf.summary.scalar('loss/Q', loss_Q),
                tf.summary.scalar('loss/pi', loss_pi),
                tf.summary.scalar('loss/alpha', loss_alpha),
                tf.summary.scalar('episode_reward', self.ep_reward)
            ]
            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        self.sess.run(tf.global_variables_initializer())

    def build(self):
        self._build_networks()
        self._build_train_ops()

    class TrainConfig(BaseTrainConfig):
        lr = 0.01
        lr_warmup_steps = 500
        lr_decay_steps = 500
        lr_decay = 0.9
        batch_size = 8
        n_steps = 50000
        buffer_capacity = 1e5
        log_interval = 100
        target_update_every_step = 50

    def train(self, config: BaseTrainConfig):
        # set up learning rate schedule
        global_step = tf.Variable(0, trainable=False, name='global_step')
        gradual_warmup_term = tf.minimum(1.0, tf.cast(global_step, tf.float32) / config.lr_warmup_steps)
        learning_rate_op = gradual_warmup_term * tf.train.exponential_decay(
            config.lr, global_step, config.lr_decay_steps, config.lr_decay, staircase=True)

        buffer = ReplayMemory(capacity=config.buffer_capacity, tuple_class=Transition)

        step = 0
        n_episode = 0
        episode_reward = 0.
        reward_history = []
        reward_averaged = []

        while step < config.n_steps:
            ob = self.env.reset()
            self.act(ob)
            done = False

            while not done:
                a = self.act(ob)
                ob_next, r, done, info = self.env.step(a)
                step += 1
                episode_reward += r

                record = Transition(self.obs_to_inputs(ob), a, r, self.obs_to_inputs(ob_next), done)
                buffer.add(record)

                ob = ob_next

                while buffer.size >= config.batch_size:
                    batch = buffer.pop(config.batch_size)
                    lr = self.sess.run(learning_rate_op, feed_dict={global_step: step})
                    feed_dict = {
                        self.lr: lr,
                        self.s: batch['s'],
                        self.a: batch['a'],
                        self.r: batch['r'],
                        self.s_next: batch['s_next'],
                        self.done: batch['done'],
                        self.ep_reward: np.mean(reward_history[-10:]) if reward_history else 0.0,  # for logging.
                    }
                    _, summ_str = self.sess.run([self.train_ops, self.merged_summary], feed_dict=feed_dict)
                    self.writer.add_summary(summ_str, step)

                if step % config.target_update_every_step == 0:
                    self.sess.run(self.update_Q_target_op)

            # One trajectory is complete!
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            episode_reward = 0.
            n_episode += 1

            if reward_history and step % config.log_interval == 0:
                # Report the performance every `every_step` steps
                logging.info(f"[episodes:{n_episode}/step:{step}], best:{np.max(reward_history)}, "
                             f"avg:{np.mean(reward_history[-10:]):.2f}:{reward_history[-5:]}, lr:{lr:.4f}")
                # self.save_checkpoint(step=step)

        self.save_checkpoint(step=step)
        logging.info(f"[FINAL] episodes: {len(reward_history)}, Max reward: {np.max(reward_history)}, "
                     f"Average reward: {np.mean(reward_history)}")

        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
        }
        plot_learning_curve(self.model_name, data_dict, xlabel='episode')
