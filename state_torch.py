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
    ], dtype=torch.float)[None, ...]

def state_maker(up, left, front, right, back, down):
    return torch.tensor([
        [up, up],
        [up, up],
        [left, left],
        [left, left],
        [front, front],
        [front, front],
        [right, right],
        [right, right],
        [back, back],
        [back, back],
        [down, down],
        [down, down],
    ], dtype=torch.float)[None, ...]

def all_solved_state():
    s11 = state_maker(4, 5, 1, 3, 6, 2)
    s12 = state_maker(5, 2, 1, 4, 6, 3)
    s13 = state_maker(2, 3, 1, 5, 6, 4)
    s14 = state_maker(3, 4, 1, 2, 6, 5)
    
    s21 = state_maker(3, 1, 2, 6, 4, 5)
    s22 = state_maker(1, 5, 2, 3, 4, 6)
    s23 = state_maker(5, 6, 2, 1, 4, 3)
    s24 = state_maker(6, 3, 2, 5, 4, 1)
    
    s31 = state_maker(1, 2, 3, 4, 5, 6)
    s32 = state_maker(2, 6, 3, 1, 5, 4)
    s33 = state_maker(6, 4, 3, 2, 5, 1)
    s34 = state_maker(4, 1, 3, 6, 5, 2)

    s41 = state_maker(3, 6, 4, 1, 2, 5)
    s42 = state_maker(6, 5, 4, 3, 2, 1)
    s43 = state_maker(5, 1, 4, 6, 2, 3)
    s44 = state_maker(1, 3, 4, 5, 2, 6)
    
    s51 = state_maker(1, 4, 5, 2, 3, 6)
    s52 = state_maker(4, 6, 5, 1, 3, 2)
    s53 = state_maker(6, 2, 5, 4, 3, 1)
    s54 = state_maker(2, 1, 5, 6, 3, 4)
    
    s61 = state_maker(4, 3, 6, 5, 1, 2)
    s62 = state_maker(3, 2, 6, 4, 1, 5)
    s63 = state_maker(2, 5, 6, 3, 1, 4)
    s64 = state_maker(5, 4, 6, 2, 1, 3)
    return torch.stack([s11, s12, s13, s14, 
                        s21, s22, s23, s24, 
                        s31, s32, s33, s34, 
                        s41, s42, s43, s44, 
                        s51, s52, s53, s54, 
                        s61, s62, s63, s64])


def next_state(state, action):

    # state.shape:(B, 12, 2)
    state = state.clone()

    # left to up
    if action == 1:
        state[:, 2:4, :] = torch.rot90(state[:, 2:4, :], 1, dims=(1, 2))
        start = torch.clone(state[:, 0:2, 0])
        state[:, 0:2, 0] = state[:, 4:6, 0]
        state[:, 4:6, 0] = state[:, 10:12, 0]
        state[:, 10:12, 0] = torch.flip(state[:, 8:10, 1], dims=(-1,))
        state[:, 8:10, 1] = torch.flip(start, dims=(-1,))

    # right to up
    elif action == 2:
        state[:, 6:8, :] = torch.rot90(state[:, 6:8, :], -1, dims=(1, 2))
        start = torch.clone(state[:, 0:2, 1])
        state[:, 0:2, 1] = state[:, 4:6, 1]
        state[:, 4:6, 1] = state[:, 10:12, 1]
        state[:, 10:12, 1] = torch.flip(state[:, 8:10, 0], dims=(-1,))
        state[:, 8:10, 0] = torch.flip(start, dims=(-1,))

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
        state[:, 0, :] = torch.flip(state[:, 2:4, 0], dims=(-1,))
        state[:, 2:4, 0] = state[:, 11, :]
        state[:, 11, :] = torch.flip(start, dims=(-1,))

    # front to right
    elif action == 6:
        state[:, 4:6, :] = torch.rot90(state[:, 4:6, :], -1, dims=(1, 2))
        start = torch.clone(state[:, 6:8, 0])
        state[:, 6:8, 0] = state[:, 1, :]
        state[:, 1, :] = torch.flip(state[:, 2:4, 1], dims=(-1,))
        state[:, 2:4, 1] = state[:, 10, :]
        state[:, 10, :] = torch.flip(start, dims=(-1,))

    # left to down
    if action == 7:
        state[:, 2:4, :] = torch.rot90(state[:, 2:4, :], -1, dims=(1, 2))
        start = torch.clone(state[:, 10:12, 0])
        state[:, 10:12, 0] = state[:, 4:6, 0]
        state[:, 4:6, 0] = state[:, 0:2, 0]
        state[:, 0:2, 0] = torch.flip(state[:, 8:10, 1], dims=(-1,))
        state[:, 8:10, 1] = torch.flip(start, dims=(-1,))

    # right to down
    if action == 8:
        state[:, 6:8, :] = torch.rot90(state[:, 6:8, :], 1, dims=(1, 2))
        start = torch.clone(state[:, 10:12, 1])
        state[:, 10:12, 1] = state[:, 4:6, 1]
        state[:, 4:6, 1] = state[:, 0:2, 1]
        state[:, 0:2, 1] = torch.flip(state[:, 8:10, 0], dims=(-1,))
        state[:, 8:10, 0] = torch.flip(start, dims=(-1,))

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
        state[:, 6:8, 1] = torch.flip(state[:, 11, :], dims=(-1,))
        state[:, 11, :] = state[:, 2:4, 0]
        state[:, 2:4, 0] = torch.flip(state[:, 0, :], dims=(-1,))
        state[:, 0, :] = start

    # front to left
    elif action == 12:
        state[:, 4:6, :] = torch.rot90(state[:, 4:6, :], 1, dims=(1, 2))
        start = torch.clone(state[:, 6:8, 0])
        state[:, 6:8, 0] = torch.flip(state[:, 10, :], dims=(-1,))
        state[:, 10, :] = state[:, 2:4, 1]
        state[:, 2:4, 1] = torch.flip(state[:, 1, :], dims=(-1,))
        state[:, 1, :] = start
    
    return state


if __name__ == '__main__':
    initial_state = solved_state().repeat(2, 1, 1)
    print('intial state:')
    print(initial_state)
    print()
    child_state = next_state(initial_state, action=2)
    print('next state:')
    print(child_state)
