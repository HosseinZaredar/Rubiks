import torch
import torch.nn as nn

class RBlock(nn.Module):
        
    def __init__(self, in_dim=64, hidden_dim=64, bn=False) -> None:
        super(RBlock, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if bn else nn.Identity()
        self.l2 = nn.Linear(hidden_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim) if bn else nn.Identity()

    def forward(self, x):
        out = self.bn1(self.l1(x))
        out = nn.functional.relu(out)
        out = self.bn2(self.l2(out))
        return out + x

class LinearModel(nn.Module):

    def __init__(self, n_rb=4, bn=False) -> None:
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(24, 256)
        self.bn1 = nn.BatchNorm1d(256) if bn else nn.Identity()
        self.layer2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64) if bn else nn.Identity()
        self.rbs = nn.Sequential(*[
            RBlock(bn=bn) for _ in range(n_rb)
        ])
        self.last_layer = nn.Linear(64, 1)
    
    def forward(self, x):
        x = (x - 1) / 5
        out = self.bn1(self.layer1(x))
        out = nn.functional.relu(out)
        out = self.bn2(self.layer2(out))
        out = nn.functional.relu(out)
        out = self.rbs(out)
        out = self.last_layer(out)
        return out


if __name__ == "__main__":
    lm = LinearModel()
    x = torch.rand(16, 24)
    y = lm(x)
    print(y.shape)