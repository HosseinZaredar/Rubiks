import torch
import pickle
import numpy as np
import os.path as osp

from dataset import RandomDataset
from algo import solve

class TestCase:
    def __init__(self, inp: np.ndarray, out) -> None:
        self.inp = inp
        self.out = out

SEED = 42
path = osp.join(".", "benchmarks")

dataset = RandomDataset(k = 20, random_seed=SEED)


for i in range(1):
    state = dataset[i].numpy().astype(np.uint8)
    solution = solve(init_state=state, init_location=None, method="BiBFS")
    print(solution)
    with open(osp.join(path, f"{i}.pkl"), "wb") as f:
        pickle.dump(TestCase(state, solution), f)