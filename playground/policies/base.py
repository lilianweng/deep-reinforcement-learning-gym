import os
import time

import numpy as np
import tensorflow as tf
from gym.utils import colorize

from playground.utils.misc import REPO_ROOT


class Policy:
    def __init__(self, env, name, training=True, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.training = training
        self.name = name

        np.random.seed(int(time.time()))

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


class BaseTFModelMixin(object):
    """Abstract object representing an Reader model.

    Code borrowed from: https://github.com/devsisters/DQN-tensorflow/blob/master/dqn/base.py
    with some modifications.
    """

    def __init__(self, model_name, saver_max_to_keep=5):
        self._saver = None
        self._saver_max_to_keep = saver_max_to_keep
        self._writer = None
        self._model_name = model_name
        self._sess = None

        # for attr in self._attrs:
        #    name = attr if not attr.startswith('_') else attr[1:]
        #    setattr(self, name, getattr(self.config, attr))

    def save_model(self, step=None):
        print(colorize(" [*] Saving checkpoints...", "green"))
        ckpt_file = os.path.join(self.checkpoint_dir, self.model_name)
        self.saver.save(self.sess, ckpt_file, global_step=step)

    def load_model(self):
        print(colorize(" [*] Loading checkpoints...", "green"))

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        print(self.checkpoint_dir, ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            print(fname)
            self.saver.restore(self.sess, fname)
            print(colorize(" [*] Load SUCCESS: %s" % fname, "green"))
            return True
        else:
            print(colorize(" [!] Load FAILED: %s" % self.checkpoint_dir, "red"))
            return False

    @property
    def checkpoint_dir(self):
        ckpt_path = os.path.join(REPO_ROOT, 'checkpoints', self.model_name)
        os.makedirs(ckpt_path, exist_ok=True)
        return ckpt_path

    @property
    def model_name(self):
        assert self._model_name, "Not a valid model name."
        return self._model_name

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=self._saver_max_to_keep)
        return self._saver

    @property
    def writer(self):
        if self._writer is None:
            writer_path = os.path.join(REPO_ROOT, "logs", self.model_name)
            os.makedirs(writer_path, exist_ok=True)
            self._writer = tf.summary.FileWriter(writer_path, self.sess.graph)
        return self._writer

    @property
    def sess(self):
        if self._sess is None:
            config = tf.ConfigProto()

            config.intra_op_parallelism_threads = 2
            config.inter_op_parallelism_threads = 2
            self._sess = tf.Session(config=config)

        return self._sess
