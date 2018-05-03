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


def sample(preds, temperature=1.0):
    # function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class ActorCriticPolicy(Policy, BaseTFModelMixin):
    def __init__(self, env, name, training=True, gamma=0.9,
                 lr=0.001, lr_decay=0.999, batch_size=32, layer_sizes=None):
        Policy.__init__(self, env, name, training=training, gamma=gamma)
        BaseTFModelMixin.__init__(self, name)

        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes or [32, 32]

    def act(self, state, return_value=False):
        # Stochastic policy
        with self.sess.as_default():
            action_proba = self.actor_proba.eval({self.states: [state]})[0]
            # print("action_proba =", action_proba)

        return np.random.choice(self.act_size, p=action_proba)

    def _scope_vars(self, scope):
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(vars) > 0
        print("Variables in scope '%s'" % scope, vars)
        return vars

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
        self.rewards = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.td_targets = tf.placeholder(tf.float32, shape=(None, ), name='td_target')

        # Actor: action probabilities
        self.actor = mlp(self.states, self.layer_sizes + [self.act_size], name='actor')
        self.actor_proba = tf.nn.softmax(self.actor)
        self.actor_vars = self._scope_vars('actor')

        # Critic: action value (Q-value)
        self.critic = mlp(self.states, [20, 1], name='critic')
        self.critic_vars = self._scope_vars('critic')

        with tf.variable_scope('critic_optimize'):
            action_ohe = tf.one_hot(self.actions, self.act_size, 1.0, 0.0, name='action_one_hot')
            pred_q = tf.reduce_sum(self.critic * action_ohe, reduction_indices=-1, name='q_acted')
            td_error = self.td_targets - pred_q

            #self.reg_critic = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.critic_vars])
            self.loss_critic = tf.reduce_mean(td_error)
            # self.loss_critic = self.loss_critic + 0.001 * self.reg_critic
            self.optim_critic = tf.train.AdamOptimizer(0.01).minimize(
                self.loss_critic, name='adam_optim_critic')

        with tf.variable_scope('actor_optimize'):
            #self.reg_actor = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.actor_vars])
            self.loss_actor = tf.reduce_mean(
                tf.stop_gradient(td_error) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.actor, labels=self.actions), name='loss_actor')
            #self.loss_actor = self.loss_actor + 0.001 * self.reg_actor

            self.optim_actor = tf.train.AdamOptimizer(0.001).minimize(
                self.loss_actor, name='adam_optim_actor')

        with tf.variable_scope('summary'):
            self.loss_critic_summ = tf.summary.scalar('loss_critic', self.loss_critic)
            self.loss_actor_summ = tf.summary.scalar('loss_actor', self.loss_actor)

            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.ep_reward_summ = tf.summary.scalar('episode_reward', self.ep_reward)
            self.merged_summary = tf.summary.merge([
                self.loss_actor_summ, self.loss_critic_summ, self.ep_reward_summ])

        self.train_ops = [self.optim_actor, self.optim_critic]

        self.sess.run(tf.global_variables_initializer())

    def train(self, n_episodes, every_episode):
        step = 0
        episode_reward = 0.
        reward_history = []
        reward_averaged = []

        lr = self.lr

        for n_episode in range(n_episodes):
            ob = self.env.reset()
            a = self.act(ob)
            done = False

            obs = []
            actions = []
            rewards = []
            td_targets = []

            while not done:
                a = self.act(ob)
                ob_next, r, done, info = self.env.step(a)
                step += 1
                episode_reward += r

                obs.append(self.obs_to_inputs(ob))
                actions.append(a)
                rewards.append(r)

                # a_next = self.act(ob_next)
                with self.sess.as_default():
                    next_value = self.critic.eval({self.states: [ob_next]})[0][0]
                td_target = r + self.gamma * next_value
                td_targets.append(td_target)

                ob = ob_next
                # a = a_next

                _, summ_str = self.sess.run(
                    [self.train_ops, self.merged_summary], feed_dict={
                        self.learning_rate: lr,
                        self.states: np.array([ob]),
                        self.actions: np.array([a]),
                        self.rewards: np.array([r]),
                        self.td_targets: np.array([td_target]),
                        self.ep_reward: reward_history[-1] if reward_history else 0.0,
                    })
                self.writer.add_summary(summ_str, step)

            # One trajectory is complete!
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            episode_reward = 0.

            # print("td_targets[-5:] =", td_targets[-5:])

            lr *= self.lr_decay

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