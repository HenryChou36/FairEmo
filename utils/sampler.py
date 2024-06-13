import torch
from torch.utils.data import Dataset, Sampler
from typing import Iterator
import numpy as np

class SplitDistSampler(Sampler):
    def __init__(self, dataset:Dataset, protect, split=2, batch_size=128, seed: int=0) ->None:
        self.dataset = dataset
        self.default_ind = np.array(range(len(dataset)))
        self.protect = np.asarray(protect)
        self.split = split
        self.batch_size = 128

        self.seed = seed
        self.epoch = 0

        self.size_per_split = int(batch_size / self.split)
        self.protect_val, self.protect_cnt = np.unique(protect, return_counts=True)
        count = 0
        for i in range(int(np.max(self.protect_cnt) / self.size_per_split)):
            valid_val = self.protect_val[self.protect_cnt > ((i+1) * self.size_per_split)]
            count += len(valid_val) - (len(valid_val) % self.split)
        self.total_size = count * self.size_per_split

    def __iter__(self) -> Iterator:
        indices = []
        rng = np.random.default_rng(seed=self.seed + self.epoch)
        group_indices = dict.fromkeys(self.protect_val)
        for val in self.protect_val:
            group_indices[val] = self.default_ind[self.protect == val]
            rng.shuffle(group_indices[val])

        for i in range(int(np.max(self.protect_cnt) / self.size_per_split)):
            valid_val = self.protect_val[self.protect_cnt > ((i+1) * self.size_per_split)]
            rng.shuffle(valid_val)
            for j in range(len(valid_val) - (len(valid_val) % self.split)):
                indices.extend(group_indices[valid_val[j]][i*self.size_per_split:(i+1)*self.size_per_split].tolist())
        return iter(indices)
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total_size
