from collections import namedtuple

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from playground.policies.base import BaseModelMixin, Policy, BaseTrainConfig
from playground.policies.memory import ReplayMemory
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import dense_nn


class PPOPolicy(Policy, BaseModelMixin):

    def __init__(self, env, name, training=True, gamma=0.99, lam=0.95,
                 actor_layers=[64, 32], critic_layers=[128, 64], clip_norm=None, **kwargs):
        Policy.__init__(self, env, name, training=training, gamma=gamma, **kwargs)
        BaseModelMixin.__init__(self, name)

        assert isinstance(self.env.action_space, Discrete), \
            "Current PPOPolicy implementation only works for discrete action space."

        self.lam = lam  # lambda for GAE.
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.clip_norm = clip_norm

    def act(self, state, **kwargs):
        probas = self.sess.run(self.actor_proba, {self.s: [state]})[0]
        action = np.random.choice(range(self.act_size), size=1, p=probas)[0]
        return action

    def _build_networks(self):
        # Define input placeholders
        self.s = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='state')
        self.a = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.s_next = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='next_state')
        self.r = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.done = tf.placeholder(tf.float32, shape=(None,), name='done_flag')

        self.old_logp_a = tf.placeholder(tf.float32, shape=(None,), name='old_logp_actor')
        self.v_target = tf.placeholder(tf.float32, shape=(None,), name='v_target')
        self.adv = tf.placeholder(tf.float32, shape=(None,), name='return')

        with tf.variable_scope('actor'):
            # Actor: action probabilities
            self.actor = dense_nn(self.s, self.actor_layers + [self.act_size], name='actor')
            self.actor_proba = tf.nn.softmax(self.actor)
            a_ohe = tf.one_hot(self.a, self.act_size, 1.0, 0.0, name='action_ohe')
            self.logp_a = tf.reduce_sum(tf.log(self.actor_proba) * a_ohe,
                                        reduction_indices=-1, name='new_logp_actor')
            self.actor_vars = self.scope_vars('actor')

        with tf.variable_scope('critic'):
            # Critic: action value (V value)
            self.critic = tf.squeeze(dense_nn(self.s, self.critic_layers + [1], name='critic'))
            self.critic_next = tf.squeeze(dense_nn(self.s_next, self.critic_layers + [1], name='critic', reuse=True))
            self.critic_vars = self.scope_vars('critic')

    def _build_train_ops(self):
        self.lr_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_actor')
        self.lr_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_critic')
        self.clip_range = tf.placeholder(tf.float32, shape=None, name='ratio_clip_range')

        with tf.variable_scope('actor_train'):
            ratio = tf.exp(self.logp_a - self.old_logp_a)
            ratio_clipped = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            loss_a = - tf.reduce_mean(tf.minimum(self.adv * ratio, self.adv * ratio_clipped))

            optim_a = tf.train.AdamOptimizer(self.lr_a)
            grads_a = optim_a.compute_gradients(loss_a, var_list=self.actor_vars)
            if self.clip_norm:
                grads_a = [(tf.clip_by_norm(g, self.clip_norm), v) for g, v in grads_a]
            self.train_op_a = optim_a.apply_gradients(grads_a)

        with tf.variable_scope('critic_train'):
            loss_c = tf.reduce_mean(tf.square(self.v_target - self.critic))

            optim_c = tf.train.AdamOptimizer(self.lr_c)
            grads_c = optim_c.compute_gradients(loss_c, var_list=self.critic_vars)
            if self.clip_norm:
                grads_c = [(tf.clip_by_norm(g, self.clip_norm), v) for g, v in grads_c]
            self.train_op_c = optim_c.apply_gradients(grads_c)

        self.train_ops = [self.train_op_a, self.train_op_c]

        with tf.variable_scope('summary'):
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')

            self.summary = [
                tf.summary.scalar('loss/adv', tf.reduce_mean(self.adv)),
                tf.summary.scalar('loss/ratio', tf.reduce_mean(ratio)),
                tf.summary.scalar('loss/loss_actor', loss_a),
                tf.summary.scalar('loss/loss_critic', loss_c),
                tf.summary.scalar('episode_reward', self.ep_reward)
            ]

            # self.summary += [tf.summary.scalar('grads/' + v.name, tf.norm(g))
            #                 for g, v in grads_a if g is not None]
            # self.summary += [tf.summary.scalar('grads/' + v.name, tf.norm(g))
            #                 for g, v in grads_c if g is not None]

            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        self.sess.run(tf.global_variables_initializer())

    def build(self):
        self._build_networks()
        self._build_train_ops()

    class TrainConfig(BaseTrainConfig):
        lr_a = 0.005
        lr_c = 0.005
        batch_size = 32
        n_iterations = 100
        n_rollout_workers = 5
        train_epoches = 5
        log_every_iteration = 10
        ratio_clip_range = 0.2
        ratio_clip_decay = True

    def _generate_rollout(self, buffer):
        # generate one trajectory.
        ob = self.env.reset()
        done = False
        rewards = []
        episode_reward = 0.0
        obs = []
        actions = []

        while not done:
            a = self.act(ob)
            ob_next, r, done, info = self.env.step(a)
            obs.append(ob)
            actions.append(a)
            rewards.append(r)
            episode_reward += r
            ob = ob_next

        # length of the episode.
        T = len(rewards)

        # compute the current log pi(a|s) and predicted v values.
        with self.sess.as_default():
            logp_a = self.logp_a.eval({self.a: np.array(actions), self.s: np.array(obs)})
            v_pred = self.critic.eval({self.s: np.array(obs)})

        # Compute TD errors
        td_errors = [rewards[t] + self.gamma * v_pred[t + 1] - v_pred[t] for t in range(T - 1)]
        td_errors += [rewards[T - 1] + self.gamma * 0.0 - v_pred[T - 1]]  # handle the terminal state.

        assert len(logp_a) == len(v_pred) == len(td_errors) == T

        # Estimate advantage backwards.
        advs = []
        adv_so_far = 0.0
        for delta in td_errors[::-1]:
            adv_so_far = delta + self.gamma * self.lam * adv_so_far
            advs.append(adv_so_far)
        advs = advs[::-1]
        assert len(advs) == T

        # add into the memory buffer
        v_targets = np.array(advs) + np.array(v_pred)
        for i, (s, a, s_next, r, old_logp_a, v_target, adv) in enumerate(zip(
                obs, actions, np.array(obs[1:] + [ob_next]), rewards,
                np.squeeze(logp_a), v_targets, advs)):
            done = float(i == T - 1)
            buffer.add(buffer.tuple_class(s, a, s_next, r, done, old_logp_a, v_target, adv))

        return episode_reward, len(advs)

    def train(self, config: TrainConfig):
        BufferRecord = namedtuple('BufferRecord', ['s', 'a', 's_next', 'r', 'done',
                                                   'old_logp_actor', 'v_target', 'adv'])
        buffer = ReplayMemory(tuple_class=BufferRecord)

        reward_history = []
        reward_averaged = []
        step = 0
        total_rec = 0

        clip = config.ratio_clip_range
        if config.ratio_clip_decay:
            clip_delta = clip / config.n_iterations
        else:
            clip_delta = 0.0

        for n_iteration in range(config.n_iterations):

            # we should have multiple rollout_workers running in parallel.
            for _ in range(config.n_rollout_workers):
                episode_reward, n_rec = self._generate_rollout(buffer)
                # One trajectory is complete.
                reward_history.append(episode_reward)
                reward_averaged.append(np.mean(reward_history[-10:]))
                total_rec += n_rec

            # now let's train the model for some steps.
            for batch in buffer.loop(config.batch_size, epoch=config.train_epoches):
                _, summ_str = self.sess.run(
                    [self.train_ops, self.merged_summary], feed_dict={
                        self.lr_a: config.lr_a,
                        self.lr_c: config.lr_c,
                        self.clip_range: clip,
                        self.s: batch['s'],
                        self.a: batch['a'],
                        self.s_next: batch['s_next'],
                        self.r: batch['r'],
                        self.done: batch['done'],
                        self.old_logp_a: batch['old_logp_actor'],
                        self.v_target: batch['v_target'],
                        self.adv: batch['adv'],
                        self.ep_reward: np.mean(reward_history[-10:]) if reward_history else 0.0,
                    })

                self.writer.add_summary(summ_str, step)
                step += 1

            clip = max(0.0, clip - clip_delta)

            if (reward_history and config.log_every_iteration and
                    n_iteration % config.log_every_iteration == 0):
                # Report the performance every `log_every_iteration` steps
                print("[iteration:{}/step:{}], best:{}, avg:{:.2f}, hist:{}, clip:{:.2f}; {} transitions.".format(
                    n_iteration, step, np.max(reward_history), np.mean(reward_history[-10:]),
                    list(map(lambda x: round(x, 2), reward_history[-5:])), clip, total_rec
                ))
                # self.save_checkpoint(step=step)

        self.save_checkpoint(step=step)

        print("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
        }
        plot_learning_curve(self.model_name, data_dict, xlabel='episode')
