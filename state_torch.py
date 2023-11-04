import torch


def solved_state():
    return torch.tensor([
        [1, 1],
        [1, 1],
        [2, 2],
        [2, 2],
        [3, 3],
        [3, 3],
        [4, 4],
        [4, 4],
        [5, 5],
        [5, 5],
        [6, 6],
        [6, 6],
    ])[None, ...]


def next_state(state, action):

    # state.shape:(B, 12, 2)
    state = state.clone()

    # left to up
    if action == 1:
        state[:, 2:4, :] = torch.rot90(state[:, 2:4, :], 1, dims=(1, 2))
        start = state[0:2, 0].clone()
        state[:, 0:2, 0] = state[:, 4:6, 0]
        state[:, 4:6, 0] = state[:, 10:12, 0]
        state[:, 10:12, 0] = torch.flip(state[:, 8:10, 1])
        state[:, 8:10, 1] = torch.flip(start)

    # right to up
    elif action == 2:
        state[:, 6:8, :] = torch.rot90(state[:, 6:8, :], -1, dims=(1, 2))
        start = torch.clone(state[:, 0:2, 1])
        state[:, 0:2, 1] = state[:, 4:6, 1]
        state[:, 4:6, 1] = state[:, 10:12, 1]
        state[:, 10:12, 1] = torch.flip(state[:, 8:10, 0])
        state[:, 8:10, 0] = torch.flip(start)

    # down to left
    elif action == 3:
        state[:, 10:12, :] = torch.rot90(state[:, 10:12, :], 1, dims=(1, 2))
        start = torch.clone(state[:, 3, :])
        state[:, 3, :] = state[:, 5, :]
        state[:, 5, :] = state[:, 7, :]
        state[:, 7, :] = state[:, 9, :]
        state[:, 9, :] = start

    # up to left
    elif action == 4:
        state[:, 0:2, :] = torch.rot90(state[:, 0:2, :], -1, dims=(1, 2))
        start = torch.clone(state[:, 2, :])
        state[:, 2, :] = state[:, 4, :]
        state[:, 4, :] = state[:, 6, :]
        state[:, 6, :] = state[:, 8, :]
        state[:, 8, :] = start

    # back to right
    elif action == 5:
        state[:, 8:10, :] = torch.rot90(state[:, 8:10, :], 1, dims=(1, 2))
        start = torch.clone(state[:, 6:8, 1])
        state[:, 6:8, 1] = state[:, 0, :]
        state[:, 0, :] = torch.flip(state[:, 2:4, 0])
        state[:, 2:4, 0] = state[:, 11, :]
        state[:, 11, :] = torch.flip(start)

    # front to right
    elif action == 6:
        state[:, 4:6, :] = torch.rot90(state[:, 4:6, :], -1, dims=(1, 2))
        start = torch.clone(state[:, 6:8, 0])
        state[:, 6:8, 0] = state[:, 1, :]
        state[:, 1, :] = torch.flip(state[:, 2:4, 1])
        state[:, 2:4, 1] = state[:, 10, :]
        state[:, 10, :] = torch.flip(start)

    # left to down
    if action == 7:
        state[:, 2:4, :] = torch.rot90(state[:, 2:4, :], -1, dims=(1, 2))
        start = torch.clone(state[:, 10:12, 0])
        state[:, 10:12, 0] = state[:, 4:6, 0]
        state[:, 4:6, 0] = state[:, 0:2, 0]
        state[:, 0:2, 0] = torch.flip(state[:, 8:10, 1])
        state[:, 8:10, 1] = torch.flip(start)

    # right to down
    if action == 8:
        state[:, 6:8, :] = torch.rot90(state[:, 6:8, :], 1, dims=(1, 2))
        start = torch.clone(state[:, 10:12, 1])
        state[:, 10:12, 1] = state[:, 4:6, 1]
        state[:, 4:6, 1] = state[:, 0:2, 1]
        state[:, 0:2, 1] = torch.flip(state[:, 8:10, 0])
        state[:, 8:10, 0] = torch.flip(start)

    # down to right
    elif action == 9:
        state[:, 10:12, :] = torch.rot90(state[:, 10:12, :], -1, dims=(1, 2))
        start = torch.clone(state[:, 9, :])
        state[:, 9, :] = state[:, 7, :]
        state[:, 7, :] = state[:, 5, :]
        state[:, 5, :] = state[:, 3, :]
        state[:, 3, :] = start

    # up to right
    elif action == 10:
        state[:, 0:2, :] = torch.rot90(state[:, 0:2, :], 1, dims=(1, 2))
        start = torch.clone(state[:, 8, :])
        state[:, 8, :] = state[:, 6, :]
        state[:, 6, :] = state[:, 4, :]
        state[:, 4, :] = state[:, 2, :]
        state[:, 2, :] = start

    # back to left
    elif action == 11:
        state[:, 8:10, :] = torch.rot90(state[:, 8:10, :], -1, dims=(1, 2))
        start = torch.clone(state[:, 6:8, 1])
        state[:, 6:8, 1] = torch.flip(state[:, 11, :])
        state[:, 11, :] = state[:, 2:4, 0]
        state[:, 2:4, 0] = torch.flip(state[:, 0, :])
        state[:, 0, :] = start

    # front to left
    elif action == 12:
        state[:, 4:6, :] = torch.rot90(state[:, 4:6, :], 1, dims=(1, 2))
        start = torch.clone(state[:, 6:8, 0])
        state[:, 6:8, 0] = torch.flip(state[:, 10, :])
        state[:, 10, :] = state[:, 2:4, 1]
        state[:, 2:4, 1] = torch.flip(state[:, 1, :])
        state[:, 1, :] = start
    
    return state


if __name__ == '__main__':
    initial_state = solved_state()
    print('intial state:')
    print(initial_state)
    print()
    child_state = next_state(initial_state, action=4)
    print('next state:')
    print(child_state)
