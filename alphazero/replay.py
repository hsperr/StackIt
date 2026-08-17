"""A bounded replay buffer of (state, pi, z, ownership) examples.

Old samples are evicted so the net keeps training on data from recent (stronger)
policies. Sampling is uniform over whatever is currently in the buffer.
"""
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def add_many(self, examples):
        self.buf.extend(examples)

    def sample(self, batch_size, device):
        n = len(self.buf)
        idx = np.random.randint(0, n, size=min(batch_size, n))
        planes = np.stack([self.buf[i][0] for i in idx])
        pis = np.stack([self.buf[i][1] for i in idx])
        zs = np.array([self.buf[i][2] for i in idx], dtype=np.float32)
        owns = np.stack([self.buf[i][3] for i in idx]).astype(np.int64)
        return (torch.from_numpy(planes).to(device),
                torch.from_numpy(pis).to(device),
                torch.from_numpy(zs).to(device),
                torch.from_numpy(owns).to(device))
