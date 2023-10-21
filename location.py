import numpy as np


matrix = np.zeros((8, 8), dtype=np.uint8)
matrix[:4, :4] = 1
matrix[:4, 4:] = 2
matrix[4:, 4:] = 1
matrix[4:, :4] = 2
np.fill_diagonal(matrix, 0)
np.fill_diagonal(np.fliplr(matrix), 3)
np.fill_diagonal(matrix[:4, 4:], 1)
np.fill_diagonal(matrix[4:, :4], 1)
np.fill_diagonal(np.fliplr(matrix[:4, :4]), 2)
np.fill_diagonal(np.fliplr(matrix[4:, 4:]), 2)


def init_location():
    return np.array([
        [[1, 2],
        [3, 4]],
        [[5, 6],
        [7, 8]],
    ], dtype=np.uint8)


def next_location(location, action):

    location = np.copy(location)

    # left to up
    if action == 1:
        location[:, :, 0] = np.rot90(location[:, :, 0], 1)

    # right to up
    elif action == 2:
        location[:, :, 1] = np.rot90(location[:, :, 1], 1)

    # down to left
    elif action == 3:
        location[:, 1, :] = np.rot90(location[:, 1, :], 1)

    # up to left
    elif action == 4:
        location[:, 0, :] = np.rot90(location[:, 0, :], 1)

    # back to right
    elif action == 5:
        location[1, :, :] = np.rot90(location[1, :, :], -1)

    # front to right
    elif action == 6:
        location[0, :, :] = np.rot90(location[0, :, :], -1)

    # left to down
    if action == 7:
        location[:, :, 0] = np.rot90(location[:, :, 0], -1)

    # right to down
    if action == 8:
        location[:, :, 1] = np.rot90(location[:, :, 1], -1)
        
    # down to right
    elif action == 9:
        location[:, 1, :] = np.rot90(location[:, 1, :], -1)

    # up to right
    elif action == 10:
        location[:, 0, :] = np.rot90(location[:, 0, :], -1)

    # back to left
    elif action == 11:
        location[1, :, :] = np.rot90(location[1, :, :], 1)

    # front to left
    elif action == 12:
        location[0, :, :] = np.rot90(location[0, :, :], 1)

    return location


def heuristic(location):
    matrix = np.zeros((8, 8), dtype=np.uint8)
    matrix[:4, :4] = 1
    matrix[:4, 4:] = 2
    matrix[4:, 4:] = 1
    matrix[4:, :4] = 2
    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(np.fliplr(matrix), 3)
    np.fill_diagonal(matrix[:4, 4:], 1)
    np.fill_diagonal(matrix[4:, :4], 1)
    np.fill_diagonal(np.fliplr(matrix[:4, :4]), 2)
    np.fill_diagonal(np.fliplr(matrix[4:, 4:]), 2)
    select_idx = location.flatten() - 1
    output_array = np.choose(select_idx, matrix)
    h = np.sum(output_array)
    return h


if __name__ == '__main__':
    location = init_location()
    for a in [1, 3]:
        location = next_location(location, action=a)
    print(location)