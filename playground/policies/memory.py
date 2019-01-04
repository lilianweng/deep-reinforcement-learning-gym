from collections import deque, namedtuple
import numpy as np
import itertools

# This is the default buffer record nametuple type.
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_next', 'done'])


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

    def loop(self, batch_size, epoch=None):
        indices = []
        ep = None
        for i in itertools.cycle(range(len(self.buffer))):
            indices.append(i)
            if i == 0:
                ep = 0 if ep is None else ep + 1
            if epoch is not None and ep == epoch:
                break

            if len(indices) == batch_size:
                yield self._reformat(indices)
                indices = []

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
