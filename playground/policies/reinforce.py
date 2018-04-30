"""
The process is pretty straightforward:

1. Initialize the policy parameter θ at random.
2. Generate one trajectory on policy πθ: S1,A1,R2,S2,A2,…,ST.
3. For t = 1, 2, ... , T:
    - Estimate the the return Gt;
    - Update policy parameters: θ <-- θ + α γ**t (Gt - v(s_t)) ∇_θ ln π_θ(At|St)

https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#reinforce
"""
import numpy as np
import tensorflow as tf
from playground.policies.base import BaseTFModelMixin, Policy
from playground.utils.misc import plot_learning_curve
from playground.utils.tf_ops import mlp


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
        # Stochastic policy
        with self.sess.as_default():
            action_proba = self.pi_softmax.eval({self.states: [state]})[0]

        return np.random.choice(self.act_size, p=action_proba)

    @property
    def act_size(self):
        return self.env.action_space.n

    @property
    def obs_size(self):
        return self.env.observation_space.sample().flatten().shape[0]

    def obs_to_inputs(self, ob):
        return ob.flatten()

    def build(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')

        # Inputs
        self.states = tf.placeholder(tf.float32, shape=(None, self.obs_size), name='state')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.returns = tf.placeholder(tf.float32, shape=(None,), name='return')

        # Build network
        self.pi = mlp(self.states, self.layer_sizes + [self.act_size], name='pi')
        self.pi_softmax = tf.nn.softmax(self.pi)
        # self.v = mlp(self.states, self.layer_sizes + [1], name='v')  # This is the baseline.

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pi, labels=self.actions)
        self.loss = tf.reduce_mean(loss * self.returns)

        trainable_vars = tf.trainable_variables()
        print("trainable_vars:", trainable_vars)
        print("model size:", np.sum([np.product(tvar.get_shape().as_list())
                                     for tvar in trainable_vars]))

        self.grads = tf.gradients(self.loss, trainable_vars)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(self.grads, trainable_vars))

        with tf.variable_scope('summary'):
            self.loss_summ = tf.summary.scalar('loss', self.loss)
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.ep_reward_summ = tf.summary.scalar('episode_reward', self.ep_reward)
            self.merged_summary = tf.summary.merge([self.loss_summ, self.ep_reward_summ])

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

            _, loss_val, summ_str = self.sess.run(
                [self.train_op, self.loss, self.merged_summary], feed_dict={
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
