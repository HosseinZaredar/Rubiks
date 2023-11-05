from torch.utils.data import Dataset, DataLoader
import os
import pickle
import random
from collections import namedtuple

from state_torch import *


TestCase = namedtuple('TestCase', ['inp', 'out'])

class RandomDataset(Dataset):

    def __init__(self, k=20, random_seed=42) -> None:
        super(RandomDataset, self).__init__()
        random.seed(random_seed)
        self.k = k

    def __getitem__(self, index):
        state = solved_state()
        k = random.randint(1, self.k)
        for _ in range(k):
            state = next_state(state, random.randint(1, 12))
        return state.squeeze()
    
    def __len__(self):
        return 4096 * 100
    

class BenchmarkDataset(Dataset):

    def __init__(self, path) -> None:
        super(BenchmarkDataset, self).__init__()
        files = os.listdir(path)
        self.data = [pickle.load(open(os.path.join(path, file), 'rb')) for file in files]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index][0]).to(torch.float), \
            torch.from_numpy(self.data[index][1]).to(torch.float)
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    bd = BenchmarkDataset(path='benchmarks')
    dl = DataLoader(bd, batch_size=len(bd))
    for x, y in dl:
        print(x, y)
    