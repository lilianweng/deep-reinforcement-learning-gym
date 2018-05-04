import os
import time
from collections import namedtuple
import numpy as np
import tensorflow as tf
from gym.utils import colorize
from collections import deque
from playground.utils.misc import REPO_ROOT

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_next', 'done'], verbose=True)


class Trajectory:
    def __init__(self):
        self.buffer = []

    def add(self, t):
        self.buffer.append(t)


class ReplayMemory:
    def __init__(self, capacity=100000, replace=False, tuple_class=Transition):
        self.buffer = []
        self.capacity = capacity
        self.replace = replace
        self.tuple_class = tuple_class
        self.fields = tuple_class._fields

    def add(self, tuple):
        """Any named tuple item."""
        assert isinstance(tuple, self.tuple_class)
        self.buffer.append(tuple)
        while self.capacity and self.size > self.capacity:
            self.buffer.pop(0)

    def _reformat(self, indices):
        # Reformat a list of Transition tuples for training.
        # indices: list<int>
        return {
            field_name: np.array([getattr(self.buffer[i], field_name) for i in indices])
            for field_name in self.fields
        }

    def sample(self, batch_size):
        assert len(self.buffer) >= batch_size
        idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=self.replace)
        return self._reformat(idxs)

    def pop(self, batch_size):
        # Pop the first `batch_size` Transition items out.
        i = min(self.size, batch_size)
        batch = self._reformat(range(i))
        self.buffer = self.buffer[i:]
        return batch

    @property
    def size(self):
        return len(self.buffer)


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


class BaseTFModelMixin:
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

    def scope_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(res) > 0
        return res

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
