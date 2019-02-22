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
        self._entropy_target = -self.act_size

    def act(self, state):
        # Discrete actions
        proba = self.sess.run(self.pi, feed_dict={self.s: [state]})[0]
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

        # Policy: predict action probabilities
        self.pi = dense_nn(self.s, pi_network_arch, output_fn=tf.nn.softmax, name='pi')
        self.pi_next = dense_nn(self.s_next, pi_network_arch, output_fn=tf.nn.softmax, name='pi',
                                reuse=True)  # a_t+1 ~ pi(.|s_t+1)
        self.pi_vars = self.scope_vars('pi')

        # Q function: we would like to learn two soft Q functions independently to
        # mitigate bias introduced during the policy improvement.
        self.Q1, self.Q1_target_next, self.update_Q1_target_op = self._build_Q_networks('Q1')
        self.Q2, self.Q2_target_next, self.update_Q2_target_op = self._build_Q_networks('Q2')

        # Temperature alpha; we use log(alpha) because alpha should > 0.
        self.log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        self.alpha = tf.exp(self.log_alpha)

    def _build_Q_networks(self, name):
        Q_network_arch = self.layer_sizes + [self.act_size]

        Q = dense_nn(self.s, Q_network_arch, name=name + '_main')  # Q(s_t, a_t)
        Q_target_next = dense_nn(self.s_next, Q_network_arch, name=name + '_target')  # Q'(s_{t+1}, a_{t+1})

        Q_vars = self.scope_vars(name + '_main')
        Q_target_vars = self.scope_vars(name + '_target')
        assert len(Q_vars) == len(Q_target_vars)
        update_Q_target_op = [v_t.assign(v) for v_t, v in zip(Q_vars, Q_target_vars)]

        return Q, Q_target_next, update_Q_target_op

    def _build_optimization_op(self, loss, vars, lr):
        optim = tf.train.AdamOptimizer(lr)
        grads = optim.compute_gradients(loss, vars)
        if self.clip_norm:
            grads = [(tf.clip_by_norm(g, self.clip_norm), v) for g, v in grads]
        train_op = optim.apply_gradients(grads)

        return train_op, grads

    def _build_train_ops(self):
        self.lr = tf.placeholder(tf.float32, shape=None, name='learning_rate')

        a_ohe = tf.one_hot(self.a, self.act_size, 1.0, 0.0)
        a_proba = tf.reduce_sum(a_ohe * self.pi, axis=1)  # pi(a_t|s_t)
        a_logp = tf.log(a_proba)

        q1 = tf.reduce_sum(a_ohe * self.Q1, axis=1)  # Q1(s_t, a_t)
        q2 = tf.reduce_sum(a_ohe * self.Q2, axis=1)  # Q2(s_t, a_t)

        with tf.variable_scope('training_Q'):
            next_q1 = tf.reduce_sum(self.pi_next * self.Q1_target_next, axis=1)  # E_{a_t+1 ~ pi} Q(s_t+1, a_t+1)
            next_q2 = tf.reduce_sum(self.pi_next * self.Q2_target_next, axis=1)  # E_{a_t+1 ~ pi} Q(s_t+1, a_t+1)
            min_next_q = tf.minimum(next_q1, next_q2)

            # Compute the regression target.
            a_entropy_next = - tf.reduce_sum(self.pi_next * tf.log(self.pi_next), axis=1)  # H(pi(a_t+1|s_t+1))
            self.soft_V_next = min_next_q + self.alpha * a_entropy_next
            y = self.r + self.gamma * self.soft_V_next * (1.0 - self.done)

            # loss func for Q is MSE
            loss_Q1 = tf.reduce_mean(tf.square(q1 - tf.stop_gradient(y)))
            loss_Q2 = tf.reduce_mean(tf.square(q2 - tf.stop_gradient(y)))
            train_Q1_op, grads_Q1 = self._build_optimization_op(loss_Q1, self.scope_vars('Q1_main'), self.lr)
            train_Q2_op, grads_Q2 = self._build_optimization_op(loss_Q2, self.scope_vars('Q2_main'), self.lr)

        with tf.variable_scope('training_pi'):
            # The policy is trained to minimize KL divergence
            loss_pi = tf.reduce_mean(self.alpha * a_logp - tf.minimum(q1, q2))
            train_pi_op, grads_pi = self._build_optimization_op(loss_pi, self.pi_vars, self.lr)

        with tf.variable_scope('training_alpha'):
            loss_alpha = - self.log_alpha * tf.reduce_mean(tf.stop_gradient(a_logp) + self._entropy_target)
            train_alpha_op, grads_alpha = self._build_optimization_op(loss_alpha, [self.log_alpha], self.lr)

        self.losses = [loss_pi, loss_Q1, loss_Q2, loss_alpha]
        self.train_ops = [train_pi_op, train_Q1_op, train_Q2_op, train_alpha_op]

        with tf.variable_scope('summary'):
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.summary = [
                tf.summary.scalar('loss_Q1', loss_Q1),
                tf.summary.scalar('loss_Q2', loss_Q2),
                tf.summary.scalar('loss_pi', loss_pi),
                tf.summary.scalar('loss_alpha', loss_alpha),
                tf.summary.scalar('learning_rate', self.lr),
                tf.summary.scalar('avg_temperature_alpha', tf.reduce_mean(self.alpha)),
                tf.summary.scalar('episode_reward', self.ep_reward)
            ]

            # def _get_grads_summary(grads, name):
            #     return [tf.summary.scalar(f'grads_{name}/' + v.name.replace(':', '_'), tf.norm(g))
            #             for g, v in grads if g is not None]
            #
            # self.summary += _get_grads_summary(grads_pi, 'pi')
            # self.summary += _get_grads_summary(grads_Q, 'Q')
            # self.summary += _get_grads_summary(grads_alpha, 'alpha')

            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        self.sess.run(tf.global_variables_initializer())
        self.update_Q_target_networks()

    def update_Q_target_networks(self):
        self.sess.run([self.update_Q1_target_op, self.update_Q2_target_op])

    def build(self):
        self._build_networks()
        self._build_train_ops()

    class TrainConfig(BaseTrainConfig):
        lr = 0.01
        lr_warmup_steps = 500
        lr_decay_steps = 500
        lr_decay = 0.9
        batch_size = 64
        n_steps = 50000
        buffer_capacity = 1e5
        log_interval = 10
        target_update_every_step = 50

    def train(self, config: BaseTrainConfig):
        # set up learning rate schedule
        # global_step = tf.Variable(0, trainable=False, name='global_step')
        # self.sess.run(tf.variables_initializer([global_step]))
        #
        # gradual_warmup_term = tf.minimum(1.0, tf.cast(global_step, tf.float32) / config.lr_warmup_steps)
        # learning_rate_op = gradual_warmup_term * tf.train.exponential_decay(
        #     config.lr, global_step, config.lr_decay_steps, config.lr_decay, staircase=True)

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
                    # lr = self.sess.run(learning_rate_op, feed_dict={global_step: step})
                    feed_dict = {
                        self.lr: config.lr,
                        self.s: batch['s'],
                        self.a: batch['a'],
                        self.r: batch['r'],
                        self.s_next: batch['s_next'],
                        self.done: batch['done'],
                        self.ep_reward: np.mean(reward_history[-10:]) if reward_history else 0.0,  # for logging.
                    }

                    # debug_ops = {'Q': self.Q, 'pi': self.pi, 'temperature_alpha': self.alpha, 'losses': self.losses}
                    # print("[DEBUG]", self.sess.run(debug_ops, feed_dict=feed_dict))

                    _, summ_str = self.sess.run([self.train_ops, self.merged_summary], feed_dict=feed_dict)
                    self.writer.add_summary(summ_str, step)

                if step % config.target_update_every_step == 0:
                    self.update_Q_target_networks()

                if step % config.log_interval == 0:
                    if reward_history:
                        max_rew = np.max(reward_history)
                        avg10_rew = np.mean(reward_history[-10:])
                    else:
                        max_rew = -np.inf
                        avg10_rew = -np.inf

                    # Report the performance every `every_step` steps
                    logging.info(f"[episodes:{n_episode}/step:{step}], best:{max_rew}, "
                                 f"avg:{avg10_rew:.2f}:{reward_history[-5:]}, lr:{config.lr:.4f}")
                    # self.save_checkpoint(step=step)

            # One trajectory is complete!
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            episode_reward = 0.
            n_episode += 1

        self.save_checkpoint(step=step)
        logging.info(f"[FINAL] episodes: {len(reward_history)}, Max reward: {np.max(reward_history)}, "
                     f"Average reward: {np.mean(reward_history)}")

        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
        }
        plot_learning_curve(self.model_name, data_dict, xlabel='episode')
