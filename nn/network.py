import torch
import torch.nn as nn

class RBlock(nn.Module):
        
    def __init__(self, in_dim=128, hidden_dim=128) -> None:
        super(RBlock, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        out = self.l1(x)
        out = nn.functional.relu(out)
        out = self.l2(out)
        return out + x

class LinearModel(nn.Module):

    def __init__(self, n_rb=4) -> None:
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(24, 512)
        self.layer2 = nn.Linear(512, 128)
        self.rbs = nn.Sequential(*[
            RBlock() for _ in range(n_rb)
        ])
        self.last_layer = nn.Linear(128, 1)
    
    def forward(self, x):
        out = self.layer1(x)
        out = nn.functional.relu(out)
        out = self.layer2(out)
        out = nn.functional.relu(out)
        out = self.rbs(out)
        return self.last_layer(out)


if __name__ == "__main__":
    lm = LinearModel()
    x = torch.rand(16, 24)
    y = lm(x)
    print(y.shape)