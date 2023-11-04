import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import numpy as np

from dataset import RandomDataset
from state_torch import solved_state, next_state
from nn.network import LinearModel


def main():
    
    dataset = RandomDataset(k=20)
    dataloader = DataLoader(dataset, batch_size=1024, num_workers=4, drop_last=True)
    model = LinearModel(n_rb=2)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'number of parameters: {params}')

    solved_s = solved_state()

    epochs = 1000
    for e in range(epochs):

        avg_loss = 0.0

        for states in dataloader:

            # calculate the value of neighbor states
            neighbor_states = []
            for action in range(1, 12+1):
                neighbor_states.append(next_state(states, action=action))
            
            neighbor_states = torch.concat(neighbor_states, dim=0)
            neighbor_states = neighbor_states.reshape(12, states.shape[0], 12, 2)
            neighbor_states = neighbor_states.permute(1, 0, 2, 3)
            neighbor_states = neighbor_states.reshape(-1, 12, 2)

            goal_mask = 1 - (torch.eq(neighbor_states, solved_s).sum(dim=(1, 2)) == 24).to(torch.float)[..., None] 

            with torch.no_grad():
                neighbor_values = nn.functional.relu(model(neighbor_states.flatten(start_dim=1))) * goal_mask
            
            best_neighbors, _ = torch.min(neighbor_values.view(-1, 12), dim=1)
            target_values = 1 + best_neighbors

            state_values = model(states.flatten(start_dim=1)).squeeze()
            loss = loss_fn(state_values, target_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        avg_loss /= len(dataloader)
        print(f'EPOCH={e}, LOSS={avg_loss:.4f}')      
        torch.save(model.state_dict(), f'checkpoints/{e}_{int(avg_loss*1000)}.pth')      

if __name__ == '__main__':
    main()