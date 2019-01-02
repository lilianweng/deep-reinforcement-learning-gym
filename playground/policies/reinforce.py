import numpy as np
import tensorflow as tf
from playground.policies.base import BaseTFModelMixin, Policy
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import dense_nn


class ReinforcePolicy(Policy, BaseTFModelMixin):
    def __init__(self, env, name, training=True, gamma=0.99,
                 lr=0.001, lr_decay=0.999, batch_size=32, layer_sizes=None,
                 baseline=False):
        Policy.__init__(self, env, name, training=training, gamma=gamma)
        BaseTFModelMixin.__init__(self, name)

        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes or [32, 32]
        self.baseline = baseline

    def act(self, state, **kwargs):
        return self.sess.run(self.sampled_actions, {self.states: [state]})

    def build(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')

        # Inputs
        self.states = tf.placeholder(tf.float32, shape=(None, self.state_dim), name='state')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.returns = tf.placeholder(tf.float32, shape=(None,), name='return')

        # Build network
        self.pi = dense_nn(self.states, self.layer_sizes + [self.act_size], name='pi_network')
        self.sampled_actions = tf.squeeze(tf.multinomial(self.pi, 1))
        self.pi_vars = self.scope_vars('pi_network')

        if self.baseline:
            # State value estimation as the baseline
            self.v = dense_nn(self.states, self.layer_sizes + [1], name='v_network')
            self.target = self.returns - self.v  # advantage

            with tf.variable_scope('v_optimize'):
                self.loss_v = tf.reduce_mean(tf.squared_difference(self.v, self.returns))
                self.optim_v = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.loss_v, name='adam_optim_v')
        else:
            self.target = tf.identity(self.returns)

        with tf.variable_scope('pi_optimize'):
            self.loss_pi = tf.reduce_mean(
                tf.stop_gradient(self.target) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pi, labels=self.actions), name='loss_pi')
            # self.optim_pi = tf.train.AdamOptimizer(self.lr)
            # self.grads_pi = self.optim_pi.compute_gradients(self.loss_pi, self.pi_vars)
            # self.train_pi_op = self.optim_pi.apply_gradients(self.grads_pi)
            self.optim_pi = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.loss_pi, name='adam_optim_pi')

        with tf.variable_scope('summary'):
            self.loss_pi_summ = tf.summary.scalar('loss_pi', self.loss_pi)

            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.ep_reward_summ = tf.summary.scalar('episode_reward', self.ep_reward)
            summ_list = [self.loss_pi_summ, self.ep_reward_summ]

            if self.baseline:
                self.loss_v_summ = tf.summary.scalar('loss_v', self.loss_v)
                summ_list.append(self.loss_v_summ)

            self.merged_summary = tf.summary.merge(summ_list)

        if self.baseline:
            self.train_ops = [self.optim_pi, self.optim_v]
        else:
            self.train_ops = [self.optim_pi]

        self.sess.run(tf.global_variables_initializer())

    def train(self, n_episodes, every_episode):
        step = 0
        episode_reward = 0.
        reward_history = []
        reward_averaged = []

        lr = self.lr

        for n_episode in range(n_episodes):
            ob = self.env.reset()
            done = False

            obs = []
            actions = []
            rewards = []
            returns = []

            while not done:
                a = self.act(ob)
                new_ob, r, done, info = self.env.step(a)
                step += 1
                episode_reward += r

                obs.append(self.obs_to_inputs(ob))
                actions.append(a)
                rewards.append(r)
                ob = new_ob

            # One trajectory is complete!
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            episode_reward = 0.
            lr *= self.lr_decay

            # Estimate returns backwards.
            return_so_far = 0.0
            for r in rewards[::-1]:
                return_so_far = self.gamma * return_so_far + r
                returns.append(return_so_far)

            returns = returns[::-1]

            _, summ_str = self.sess.run(
                [self.train_ops, self.merged_summary], feed_dict={
                    self.learning_rate: lr,
                    self.states: np.array(obs),
                    self.actions: np.array(actions),
                    self.returns: np.array(returns),
                    self.ep_reward: reward_history[-1],
                })
            self.writer.add_summary(summ_str, step)

            if reward_history and every_episode and n_episode % every_episode == 0:
                # Report the performance every `every_step` steps
                print("[episodes:{}/step:{}], best:{}, avg:{:.2f}:{}, lr:{:.4f}".format(
                    n_episode, step, np.max(reward_history),
                    np.mean(reward_history[-10:]), reward_history[-5:],
                    lr,
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
