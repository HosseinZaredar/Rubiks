import numpy as np


def init_state():
    return np.array([
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
    ], dtype=np.uint8)


def next_state(state, action):

    state = np.copy(state)

    # left to up
    if action == 1:
        state[2:4, :] = np.rot90(state[2:4, :], 1)
        out = np.copy(state[0:2, 0])
        state[0:2, 0] = state[4:6, 0]
        state[4:6, 0] = state[10:12, 0]
        state[10:12, 0] = np.flip(state[8:10, 1])
        state[8:10, 1] = np.flip(out)

    # right to up
    elif action == 2:
        state[6:8, :] = np.rot90(state[6:8, :], -1)
        out = np.copy(state[0:2, 1])
        state[0:2, 1] = state[4:6, 1]
        state[4:6, 1] = state[10:12, 1]
        state[10:12, 1] = np.flip(state[8:10, 0])
        state[8:10, 0] = np.flip(out)

    # down to left
    elif action == 3:
        state[10:12, :] = np.rot90(state[10:12, :], 1)
        out = np.copy(state[3, :])
        state[3, :] = state[5, :]
        state[5, :] = state[7, :]
        state[7, :] = state[9, :]
        state[9, :] = out

    # up to left
    elif action == 4:
        state[0:2, :] = np.rot90(state[0:2, :], -1)
        out = np.copy(state[2, :])
        state[2, :] = state[4, :]
        state[4, :] = state[6, :]
        state[6, :] = state[8, :]
        state[8, :] = out

    # back to right
    elif action == 5:
        state[8:10, :] = np.rot90(state[8:10, :], 1)
        out = np.copy(state[6:8, 1])
        state[6:8, 1] = state[0, :]
        state[0, :] = np.flip(state[2:4, 0])
        state[2:4, 0] = state[11, :]
        state[11, :] = np.flip(out)

    # front to right
    elif action == 6:
        state[4:6, :] = np.rot90(state[4:6, :], -1)
        out = np.copy(state[6:8, 0])
        state[6:8, 0] = state[1, :]
        state[1, :] = np.flip(state[2:4, 1])
        state[2:4, 1] = state[10, :]
        state[10, :] = np.flip(out)

    # left to down
    if action == 7:
        state[2:4, :] = np.rot90(state[2:4, :], -1)
        out = np.copy(state[10:12, 0])
        state[10:12, 0] = state[4:6, 0]
        state[4:6, 0] = state[0:2, 0]
        state[0:2, 0] = np.flip(state[8:10, 1])
        state[8:10, 1] = np.flip(out)

    # right to down
    if action == 8:
        state[6:8, :] = np.rot90(state[6:8, :], 1)
        out = np.copy(state[10:12, 1])
        state[10:12, 1] = state[4:6, 1]
        state[4:6, 1] = state[0:2, 1]
        state[0:2, 1] = np.flip(state[8:10, 0])
        state[8:10, 0] = np.flip(out)

    # down to right
    elif action == 9:
        state[10:12, :] = np.rot90(state[10:12, :], -1)
        out = np.copy(state[9, :])
        state[9, :] = state[7, :]
        state[7, :] = state[5, :]
        state[5, :] = state[3, :]
        state[3, :] = out

    # up to right
    elif action == 10:
        state[0:2, :] = np.rot90(state[0:2, :], 1)
        out = np.copy(state[8, :])
        state[8, :] = state[6, :]
        state[6, :] = state[4, :]
        state[4, :] = state[2, :]
        state[2, :] = out

    # back to left
    elif action == 11:
        state[8:10, :] = np.rot90(state[8:10, :], -1)
        out = np.copy(state[6:8, 1])
        state[6:8, 1] = np.flip(state[11, :])
        state[11, :] = state[2:4, 0]
        state[2:4, 0] = np.flip(state[0, :])
        state[0, :] = out

    # front to left
    elif action == 12:
        state[4:6, :] = np.rot90(state[4:6, :], 1)
        out = np.copy(state[6:8, 0])
        state[6:8, 0] = np.flip(state[10, :])
        state[10, :] = state[2:4, 1]
        state[2:4, 1] = np.flip(state[1, :])
        state[1, :] = out
    
    return state


if __name__ == '__main__':
    
    state = init_state()

    # testing
    for i in range(12):
        for j in range(4):
            state = next_state(state, action=i+1)

    print(state)