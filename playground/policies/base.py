import os

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
from gym.utils import colorize

from playground.utils.misc import Config
from playground.utils.misc import REPO_ROOT


class TrainConfig(Config):
    lr = 0.001
    n_steps = 10000
    warmup_steps = 5000
    batch_size = 64
    log_every_step = 1000

    # give an extra bonus if done; only needed for certain tasks.
    done_reward = None


class Policy:
    def __init__(self, env, name, training=True, gamma=0.99, deterministic=False):
        self.env = env
        self.gamma = gamma
        self.training = training
        self.name = name

        if deterministic:
            np.random.seed(1)
            tf.set_random_seed(1)

    @property
    def act_size(self):
        # number of options of an action; this only makes sense for discrete actions.
        if isinstance(self.env.action_space, Discrete):
            return self.env.action_space.n
        else:
            return None

    @property
    def act_dim(self):
        # dimension of an action; this only makes sense for continuous actions.
        if isinstance(self.env.action_space, Box):
            return list(self.env.action_space.shape)
        else:
            return []

    @property
    def state_dim(self):
        # dimension of a state.
        return list(self.env.observation_space.shape)

    def obs_to_inputs(self, ob):
        return ob.flatten()

    def act(self, state, **kwargs):
        pass

    def build(self):
        pass

    def train(self, *args, **kwargs):
        pass

    def evaluate(self, n_episodes):
        reward_history = []
        reward = 0.

        for i in range(n_episodes):
            ob = self.env.reset()
            done = False
            while not done:
                a = self.act(ob)
                new_ob, r, done, _ = self.env.step(a)
                self.env.render()
                reward += r
                ob = new_ob

            reward_history.append(reward)
            reward = 0.

        print("Avg. reward over {} episodes: {:.4f}".format(n_episodes, np.mean(reward_history)))


class BaseModelMixin:
    """Abstract object representing an tensorflow model that can be easily saved/loaded.
    Modified based on https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/base.py
    """

    def __init__(self, model_name, tf_sess_config=None):
        self._saver = None
        self._writer = None
        self._model_name = model_name
        self._sess = None

        if tf_sess_config is None:
            tf_sess_config = {
                'allow_soft_placement': True,
                'intra_op_parallelism_threads': 8,
                'inter_op_parallelism_threads': 4,
            }
        self.tf_sess_config = tf_sess_config

    def scope_vars(self, scope, only_trainable=True):
        collection = tf.GraphKeys.TRAINABLE_VARIABLES if only_trainable else tf.GraphKeys.VARIABLES
        variables = tf.get_collection(collection, scope=scope)
        assert len(variables) > 0
        print(f"Variables in scope '{scope}':")
        for v in variables:
            print("\t" + str(v))
        return variables

    def get_variable_values(self):
        t_vars = tf.trainable_variables()
        vals = self.sess.run(t_vars)
        return {v.name: value for v, value in zip(t_vars, vals)}

    def save_checkpoint(self, step=None):
        print(colorize(" [*] Saving checkpoints...", "green"))
        ckpt_file = os.path.join(self.checkpoint_dir, self.model_name)
        self.saver.save(self.sess, ckpt_file, global_step=step)

    def load_checkpoint(self):
        print(colorize(" [*] Loading checkpoints...", "green"))
        ckpt_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        print(self.checkpoint_dir)
        print("ckpt_path:", ckpt_path)

        if ckpt_path:
            # self._saver = tf.train.import_meta_graph(ckpt_path + '.meta')
            self.saver.restore(self.sess, ckpt_path)
            print(colorize(" [*] Load SUCCESS: %s" % ckpt_path, "green"))
            return True
        else:
            print(colorize(" [!] Load FAILED: %s" % self.checkpoint_dir, "red"))
            return False

    def _get_dir(self, dir_name):
        path = os.path.join(REPO_ROOT, dir_name, self.model_name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def log_dir(self):
        return self._get_dir('logs')

    @property
    def checkpoint_dir(self):
        return self._get_dir('checkpoints')

    @property
    def model_dir(self):
        return self._get_dir('models')

    @property
    def tb_dir(self):
        # tensorboard
        return self._get_dir('tb')

    @property
    def model_name(self):
        assert self._model_name, "Not a valid model name."
        return self._model_name

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=5)
        return self._saver

    @property
    def writer(self):
        if self._writer is None:
            self._writer = tf.summary.FileWriter(self.tb_dir, self.sess.graph)
        return self._writer

    @property
    def sess(self):
        if self._sess is None:
            config = tf.ConfigProto(**self.tf_sess_config)
            self._sess = tf.Session(config=config)

        return self._sess
