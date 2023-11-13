import torch
import pickle
import numpy as np
import os.path as osp
from collections import namedtuple

from dataset import RandomDataset
from algo import solve

TestCase = namedtuple('TestCase', ['inp', 'out'])

SEED = 42
path = osp.join(".", "benchmarks_multigoals")

dataset = RandomDataset(k=20, random_seed=SEED)


for i in range(512):
    state = dataset[i].numpy().astype(np.uint8)
    solution = solve(init_state=state, init_location=None, method="BiBFS")
    print(solution)
    print()
    with open(osp.join(path, f"{i}.pkl"), "wb") as f:
        pickle.dump(TestCase(state, np.array(len(solution))), f)