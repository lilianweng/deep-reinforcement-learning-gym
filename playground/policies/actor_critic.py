from collections import namedtuple

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from playground.policies.base import BaseTFModelMixin, Policy, ReplayMemory, Config
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import dense_nn


class ActorCriticPolicy(Policy, BaseTFModelMixin):

    def __init__(self, env, name, training=True, gamma=0.9, layer_sizes=None, clip_norm=None, **kwargs):
        Policy.__init__(self, env, name, training=training, gamma=gamma, **kwargs)
        BaseTFModelMixin.__init__(self, name)

        assert isinstance(self.env.action_space, Discrete), \
            "Current implementation only works for discrete action space."

        self.layer_sizes = [64] if layer_sizes is None else layer_sizes
        self.clip_norm = clip_norm

    def act(self, state, eps=0.1):
        # Discrete actions
        if self.training and np.random.random() < eps:
            return self.env.action_space.sample()

        # return self.sess.run(self.sampled_actions, {self.states: [state]})
        proba = self.sess.run(self.actor_proba, {self.s: [state]})[0]
        return max(range(self.act_size), key=lambda i: proba[i])

    def _build_networks(self):
        # Define input placeholders
        self.s = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='state')
        self.a = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.r = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.td_target = tf.placeholder(tf.float32, shape=(None,), name='td_target')

        # Actor: action probabilities
        self.actor = dense_nn(self.s, self.layer_sizes + [self.act_size], name='actor')
        self.sampled_actions = tf.squeeze(tf.multinomial(self.actor, 1))
        self.actor_proba = tf.nn.softmax(self.actor)
        self.actor_vars = self.scope_vars('actor')

        # Critic: action value (V value)
        self.critic = dense_nn(self.s, self.layer_sizes + [1], name='critic')
        self.critic_vars = self.scope_vars('critic')

    def _build_train_ops(self):
        self.learning_rate_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_c')
        self.learning_rate_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_a')

        action_ohe = tf.one_hot(self.a, self.act_size, 1.0, 0.0, name='action_one_hot')
        self.pred_value = tf.reduce_sum(
            self.critic * action_ohe, reduction_indices=-1, name='q_acted')
        self.td_errors = self.td_target - tf.reshape(self.pred_value, [-1])

        with tf.variable_scope('critic_train'):
            # self.reg_c = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.critic_vars])
            self.loss_c = tf.reduce_mean(tf.square(self.td_errors))  # + 0.001 * self.reg_c
            self.optim_c = tf.train.AdamOptimizer(self.learning_rate_c)
            self.grads_c = self.optim_c.compute_gradients(self.loss_c, self.critic_vars)
            if self.clip_norm:
                self.grads_c = [(tf.clip_by_norm(grad, self.clip_norm), var)
                                for grad, var in self.grads_c]

            self.train_op_c = self.optim_c.apply_gradients(self.grads_c)

        with tf.variable_scope('actor_train'):
            # self.reg_a = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.actor_vars])
            # self.entropy_a =- tf.reduce_sum(self.actor * tf.log(self.actor))
            self.loss_a = tf.reduce_mean(
                tf.stop_gradient(self.td_errors) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.actor, labels=self.a),
                name='loss_actor')  # + 0.001 * self.reg_a
            self.optim_a = tf.train.AdamOptimizer(self.learning_rate_a)
            self.grads_a = self.optim_a.compute_gradients(self.loss_a, self.actor_vars)
            if self.clip_norm:
                self.grads_a = [(tf.clip_by_norm(grad, self.clip_norm), var)
                                for grad, var in self.grads_a]

            self.train_op_a = self.optim_a.apply_gradients(self.grads_a)

        with tf.variable_scope('summary'):
            self.grads_a_summ = [tf.summary.scalar('grads/a_' + var.name, tf.norm(grad)) for
                                 grad, var in self.grads_a if grad is not None]
            self.grads_c_summ = [tf.summary.scalar('grads/c_' + var.name, tf.norm(grad)) for
                                 grad, var in self.grads_c if grad is not None]
            self.loss_c_summ = tf.summary.scalar('loss/critic', self.loss_c)
            self.loss_a_summ = tf.summary.scalar('loss/actor', self.loss_a)

            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.ep_reward_summ = tf.summary.scalar('episode_reward', self.ep_reward)

            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        self.train_ops = [self.train_op_a, self.train_op_c]

        self.sess.run(tf.global_variables_initializer())

    def build(self):
        self._build_networks()
        self._build_train_ops()

    class TrainConfig(Config):
        lr_a = 0.02
        lr_a_decay = 0.995
        lr_c = 0.01
        lr_c_decay = 0.995
        batch_size = 32
        n_episodes = 800
        annealing_episodes = 720
        log_every_episode = 10
        done_rewards = -100
        # for epsilon-greedy exploration
        epsilon = 1.0
        epsilon_final = 0.05

    def train(self, n_episodes, config: TrainConfig):
        BufferRecord = namedtuple('Record', ['s', 'a', 'r', 'td_target'])
        buffer = ReplayMemory(tuple_class=BufferRecord)

        step = 0
        episode_reward = 0.
        reward_history = []
        reward_averaged = []

        lr_c = config.lr_c
        lr_a = config.lr_a

        eps = config.epsilon
        annealing_episodes = config.annealing_episodes or n_episodes
        eps_drop = (eps - config.epsilon_final) / annealing_episodes
        print("eps_drop:", eps_drop)

        for n_episode in range(n_episodes):
            ob = self.env.reset()
            self.act(ob, eps)
            done = False

            while not done:
                a = self.act(ob, eps)
                ob_next, r, done, info = self.env.step(a)
                step += 1
                episode_reward += r

                if done:
                    next_state_value = config.done_rewards or 0.0
                else:
                    with self.sess.as_default():
                        next_state_value = self.critic.eval({
                            self.s: [self.obs_to_inputs(ob_next)]})[0][0]

                td_target = r + self.gamma * next_state_value
                buffer.add(BufferRecord(self.obs_to_inputs(ob), a, r, td_target))
                ob = ob_next

                while buffer.size >= config.batch_size:
                    batch = buffer.pop(config.batch_size)
                    _, summ_str = self.sess.run(
                        [self.train_ops, self.merged_summary], feed_dict={
                            self.learning_rate_c: lr_c,
                            self.learning_rate_a: lr_a,
                            self.s: batch['s'],
                            self.a: batch['a'],
                            self.r: batch['r'],
                            self.td_target: batch['td_target'],
                            self.ep_reward: reward_history[-1] if reward_history else 0.0,
                        })
                    self.writer.add_summary(summ_str, step)

            # One trajectory is complete!
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            episode_reward = 0.

            lr_c *= config.lr_c_decay
            lr_a *= config.lr_a_decay
            if eps > config.epsilon_final:
                eps -= eps_drop

            if (reward_history and config.log_every_episode and
                    n_episode % config.log_every_episode == 0):
                # Report the performance every `every_step` steps
                print(
                    "[episodes:{}/step:{}], best:{}, avg:{:.2f}:{}, lr:{:.4f}|{:.4f} eps:{:.4f}".format(
                        n_episode, step, np.max(reward_history),
                        np.mean(reward_history[-10:]), reward_history[-5:],
                        lr_c, lr_a, eps,
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
