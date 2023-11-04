from torch.utils.data import Dataset
from state_torch import *
import random

class RandomDataset(Dataset):

    def __init__(self, k=20) -> None:
        super(RandomDataset, self).__init__()
        self.k = k

    def __getitem__(self, index):
        state = solved_state()
        k = random.randint(1, self.k)
        for _ in range(k):
            state = next_state(state, random.randint(1, 12))
        return state.squeeze()
    
    def __len__(self):
        return 1024 * 100