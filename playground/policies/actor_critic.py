from collections import namedtuple

import numpy as np
import tensorflow as tf

from playground.policies.base import BaseTFModelMixin, Policy, ReplayMemory
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import dense_nn

Record = namedtuple('Record', ['s', 'a', 'r', 'td_target'])


class ActorCriticPolicy(Policy, BaseTFModelMixin):
    def __init__(self, env, name, training=True, gamma=0.9,
                 lr_a=0.02, lr_a_decay=0.995,
                 lr_c=0.01, lr_c_decay=0.995,
                 epsilon=1.0,
                 epsilon_final=0.05,
                 batch_size=32,
                 layer_sizes=None,
                 grad_clip_norm=None,
                 **kwargs):
        Policy.__init__(self, env, name, training=training, gamma=gamma, **kwargs)
        BaseTFModelMixin.__init__(self, name)

        self.lr_a = lr_a
        self.lr_a_decay = lr_a_decay
        self.lr_c = lr_c
        self.lr_c_decay = lr_c_decay
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final

        self.batch_size = batch_size
        self.layer_sizes = [64] if layer_sizes is None else layer_sizes
        self.grad_clip_norm = grad_clip_norm

        self.memory = ReplayMemory(tuple_class=Record)

    def act(self, state, epsilon=0.1):
        if self.training and np.random.random() < epsilon:
            return self.env.action_space.sample()

        # return self.sess.run(self.sampled_actions, {self.states: [state]})
        proba = self.sess.run(self.actor_proba, {self.states: [state]})[0]
        return max(range(self.act_size), key=lambda i: proba[i])

    @property
    def act_size(self):
        return self.env.action_space.n

    @property
    def obs_size(self):
        return self.env.observation_space.sample().flatten().shape[0]

    def obs_to_inputs(self, ob):
        return ob.flatten()

    def build(self):
        self.learning_rate_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_c')
        self.learning_rate_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_a')

        # Inputs
        self.states = tf.placeholder(tf.float32, shape=(None, self.obs_size), name='state')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.rewards = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.td_targets = tf.placeholder(tf.float32, shape=(None,), name='td_target')

        # Actor: action probabilities
        self.actor = dense_nn(self.states, self.layer_sizes + [self.act_size], name='actor')
        self.sampled_actions = tf.squeeze(tf.multinomial(self.actor, 1))
        self.actor_proba = tf.nn.softmax(self.actor)
        self.actor_vars = self.scope_vars('actor')

        # Critic: action value (Q-value)
        self.critic = dense_nn(self.states, self.layer_sizes + [1], name='critic')
        self.critic_vars = self.scope_vars('critic')

        action_ohe = tf.one_hot(self.actions, self.act_size, 1.0, 0.0, name='action_one_hot')
        self.pred_value = tf.reduce_sum(self.critic * action_ohe, reduction_indices=-1,
                                        name='q_acted')
        self.td_errors = self.td_targets - tf.reshape(self.pred_value, [-1])

        with tf.variable_scope('critic_train'):
            # self.reg_c = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.critic_vars])
            self.loss_c = tf.reduce_mean(tf.square(self.td_errors))  # + 0.001 * self.reg_c
            self.optim_c = tf.train.AdamOptimizer(self.learning_rate_c)
            self.grads_c = self.optim_c.compute_gradients(self.loss_c, self.critic_vars)
            if self.grad_clip_norm:
                self.grads_c = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
                                for grad, var in self.grads_c]

            self.train_op_c = self.optim_c.apply_gradients(self.grads_c)

        with tf.variable_scope('actor_train'):
            # self.reg_a = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.actor_vars])
            # self.entropy_a =- tf.reduce_sum(self.actor * tf.log(self.actor))
            self.loss_a = tf.reduce_mean(
                tf.stop_gradient(self.td_errors) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.actor, labels=self.actions),
                name='loss_actor')  # + 0.001 * self.reg_a
            self.optim_a = tf.train.AdamOptimizer(self.learning_rate_a)
            self.grads_a = self.optim_a.compute_gradients(self.loss_a, self.actor_vars)
            if self.grad_clip_norm:
                self.grads_a = [(tf.clip_by_norm(grad, self.grad_clip_norm), var)
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

    def train(self, n_episodes, annealing_episodes=None, every_episode=None, done_rewards=None):
        step = 0
        episode_reward = 0.
        reward_history = []
        reward_averaged = []

        lr_c = self.lr_c
        lr_a = self.lr_a

        eps = self.epsilon
        annealing_episodes = annealing_episodes or n_episodes
        eps_drop = (eps - self.epsilon_final) / annealing_episodes
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
                    next_state_value = done_rewards or 0.0
                else:
                    with self.sess.as_default():
                        next_state_value = self.critic.eval({
                            self.states: [self.obs_to_inputs(ob_next)]})[0][0]

                td_target = r + self.gamma * next_state_value
                self.memory.add(Record(self.obs_to_inputs(ob), a, r, td_target))
                ob = ob_next

                while self.memory.size >= self.batch_size:
                    batch = self.memory.pop(self.batch_size)
                    _, summ_str = self.sess.run(
                        [self.train_ops, self.merged_summary], feed_dict={
                            self.learning_rate_c: lr_c,
                            self.learning_rate_a: lr_a,
                            self.states: batch['s'],
                            self.actions: batch['a'],
                            self.rewards: batch['r'],
                            self.td_targets: batch['td_target'],
                            self.ep_reward: reward_history[-1] if reward_history else 0.0,
                        })
                    self.writer.add_summary(summ_str, step)

            # One trajectory is complete!
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            episode_reward = 0.

            lr_c *= self.lr_c_decay
            lr_a *= self.lr_a_decay
            if eps > self.epsilon_final:
                eps -= eps_drop

            if reward_history and every_episode and n_episode % every_episode == 0:
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
