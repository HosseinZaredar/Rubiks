import torch
import os
import re
from torch.utils.data import DataLoader
from collections import namedtuple
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import copy
import argparse

from dataset import RandomDataset, BenchmarkDataset
from state_torch import solved_state, next_state
from nn.network import LinearModel

TestCase = namedtuple('TestCase', ['inp', 'out'])


def main(resume=False, tensorboard=False):

    device = torch.device('cpu')
    
    checkpoints_dir = "checkpoints"
    benchmarks_dir = 'benchmarks'
    dataset = RandomDataset(k=20)
    dataloader = DataLoader(dataset, batch_size=1024, num_workers=4, drop_last=True)

    benchmark_dataset = BenchmarkDataset(path=benchmarks_dir)
    benchmark_dataloader = DataLoader(benchmark_dataset, batch_size=len(benchmark_dataset))

    model = LinearModel(n_rb=2).to(device)

    epoch_start = 1

    if resume:
        checkpoint = find_last(path=checkpoints_dir)
        if checkpoint:
            model.load_state_dict(torch.load(checkpoint))
        checkpoint_name = os.path.split(checkpoint)[-1]
        epoch_start = int(checkpoint_name.split('_')[1])


    if tensorboard:
        from datetime import datetime
        from torch.utils.tensorboard import SummaryWriter
        run_name = f'{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'
        writer = SummaryWriter(os.path.join('tensorboard', run_name))

    optimizer = Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'number of parameters: {params}')

    solved_s = solved_state().to(device)

    epochs = 1000
    for e in range(epoch_start, epochs + 1):

        avg_loss = 0.0

        for states in dataloader:

            states = states.to(device)

            # calculate the value of neighbor states
            neighbor_states = []
            for action in range(1, 12+1):
                neighbor_states.append(next_state(states, action=action))
            
            neighbor_states = torch.concat(neighbor_states, dim=0)
            neighbor_states = neighbor_states.reshape(12, states.shape[0], 12, 2)
            neighbor_states = neighbor_states.permute(1, 0, 2, 3)
            neighbor_states = neighbor_states.reshape(-1, 12, 2)
            neighbor_states = neighbor_states.to(device)

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


        model.eval()
        benchmark_states, benchmark_target = next(iter(benchmark_dataloader))
        benchmark_states = benchmark_states.to(device)
        benchmark_target = benchmark_target.to(device)
        with torch.no_grad():
            benchmark_pred = model(benchmark_states.flatten(start_dim=1)).squeeze()
        error = loss_fn(benchmark_pred, benchmark_target)
        model.train()

        avg_loss /= len(dataloader)
        print(f'EPOCH={e:05d}, TRAIN LOSS={avg_loss:.4f}, TEST ERROR={error:.4f}')      
        torch.save(model.state_dict(), f'{checkpoints_dir}/epoch_{e:05d}_{str(int(avg_loss*1000)).zfill(3)}.pth')

        if tensorboard:
            writer.add_scalar('metrics/train_loss', avg_loss, e) 
            writer.add_scalar('metrics/test_error', error, e) 

def find_last(path):
    files = os.listdir(path)
    pattern = r"\d+_\d{3}\.pth"
    files = [f for f in files if re.search(pattern, f)]
    files.sort(reverse=True)
    if files:
        return os.path.join(path, files[0])
    return None


if __name__ == '__main__':

    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--tensorboard', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    main(resume=args.resume, tensorboard=args.tensorboard)