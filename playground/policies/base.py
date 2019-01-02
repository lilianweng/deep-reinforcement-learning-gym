import os
from collections import deque, namedtuple
from gym.spaces import Box, Discrete
import numpy as np
import tensorflow as tf
from gym.utils import colorize
from playground.utils.misc import Config
from playground.utils.misc import REPO_ROOT

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_next', 'done'])


class TrainConfig(Config):
    lr = 0.001
    n_steps = 10000
    warmup_steps = 5000
    batch_size = 64
    log_every_step = 1000

    # give an extra bonus if done; only needed for certain tasks.
    done_rewards = None


class ReplayMemory:
    def __init__(self, capacity=100000, replace=False, tuple_class=Transition):
        self.buffer = []
        self.capacity = capacity
        self.replace = replace
        self.tuple_class = tuple_class
        self.fields = tuple_class._fields

    def add(self, record):
        """Any named tuple item."""
        if isinstance(record, self.tuple_class):
            self.buffer.append(record)
        elif isinstance(record, list):
            self.buffer += record

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


class ReplayTrajMemory:
    def __init__(self, capacity=100000, step_size=16):
        self.buffer = deque(maxlen=capacity)
        self.step_size = step_size

    def add(self, traj):
        # traj (list<Transition>)
        if len(traj) >= self.step_size:
            self.buffer.append(traj)

    def sample(self, batch_size):
        traj_idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=True)
        batch_data = {field_name: [] for field_name in Transition._fields}

        for traj_idx in traj_idxs:
            i = np.random.randint(0, len(self.buffer[traj_idx]) + 1 - self.step_size)
            transitions = self.buffer[traj_idx][i: i + self.step_size]

            for field_name in Transition._fields:
                batch_data[field_name] += [getattr(t, field_name) for t in transitions]

        assert all(len(v) == batch_size * self.step_size for v in batch_data.values())
        return {k: np.array(v) for k, v in batch_data.items()}

    @property
    def size(self):
        return len(self.buffer)

    @property
    def transition_size(self):
        return sum(map(len, self.buffer))


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

    def get_vars(self, scope, only_trainable=True):
        collection = tf.GraphKeys.TRAINABLE_VARIABLES if only_trainable else tf.GraphKeys.VARIABLES
        variables = tf.get_collection(collection, scope=scope)
        print(scope, variables)
        assert len(variables) > 0
        return variables

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

    def scope_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        assert len(res) > 0
        print("Variables in scope '%s'" % scope)
        for v in res:
            print("\t" + str(v))
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
