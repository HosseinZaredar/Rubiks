import numpy as np


class Model():
    def __init__(self):
        self.state = np.array([
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

    def get_state(self):
        return self.state
    
    def transition(self, action):

        # left to up
        if action == 1:
            self.state[2:4, :] = np.rot90(self.state[2:4, :], 1)
            out = np.copy(self.state[0:2, 0])
            self.state[0:2, 0] = self.state[4:6, 0]
            self.state[4:6, 0] = self.state[10:12, 0]
            self.state[10:12, 0] = np.flip(self.state[8:10, 1])
            self.state[8:10, 1] = np.flip(out)

        # right to up
        elif action == 2:
            self.state[6:8, :] = np.rot90(self.state[6:8, :], -1)
            out = np.copy(self.state[0:2, 1])
            self.state[0:2, 1] = self.state[4:6, 1]
            self.state[4:6, 1] = self.state[10:12, 1]
            self.state[10:12, 1] = np.flip(self.state[8:10, 0])
            self.state[8:10, 0] = np.flip(out)

        # down to left
        elif action == 3:
            self.state[10:12, :] = np.rot90(self.state[10:12, :], 1)
            out = np.copy(self.state[3, :])
            self.state[3, :] = self.state[5, :]
            self.state[5, :] = self.state[7, :]
            self.state[7, :] = self.state[9, :]
            self.state[9, :] = out

        # up to left
        elif action == 4:
            self.state[0:2, :] = np.rot90(self.state[0:2, :], 1)
            out = np.copy(self.state[2, :])
            self.state[2, :] = np.flip(self.state[8, :])
            self.state[8, :] = self.state[6, :]
            self.state[6, :] = self.state[4, :]
            self.state[4, :] = out

        # back to right
        elif action == 5:
            self.state[8:10, :] = np.rot90(self.state[8:10, :], 1)
            out = np.copy(self.state[6:8, 1])
            self.state[6:8, 1] = self.state[0, :]
            self.state[0, :] = np.flip(self.state[2:4, 0])
            self.state[2:4, 0] = self.state[11, :]
            self.state[11, :] = np.flip(out)

        # front to right
        elif action == 6:
            self.state[4:6, :] = np.rot90(self.state[4:6, :], -1)
            out = np.copy(self.state[6:8, 0])
            self.state[6:8, 0] = self.state[1, :]
            self.state[1, :] = np.flip(self.state[2:4, 1])
            self.state[2:4, 1] = self.state[10, :]
            self.state[10, :] = np.flip(out)

        # left to down
        if action == 7:
            self.state[2:4, :] = np.rot90(self.state[2:4, :], -1)
            out = np.copy(self.state[10:12, 0])
            self.state[10:12, 0] = self.state[4:6, 0]
            self.state[4:6, 0] = self.state[0:2, 0]
            self.state[0:2, 0] = np.flip(self.state[8:10, 1])
            self.state[8:10, 1] = np.flip(out)

        # right to down
        if action == 8:
            self.state[6:8, :] = np.rot90(self.state[6:8, :], 1)
            out = np.copy(self.state[10:12, 1])
            self.state[10:12, 1] = self.state[4:6, 1]
            self.state[4:6, 1] = self.state[0:2, 1]
            self.state[0:2, 1] = np.flip(self.state[8:10, 0])
            self.state[8:10, 0] = np.flip(out)

        # down to right
        elif action == 9:
            self.state[10:12, :] = np.rot90(self.state[10:12, :], -1)
            out = np.copy(self.state[9, :])
            self.state[9, :] = self.state[7, :]
            self.state[7, :] = self.state[5, :]
            self.state[5, :] = self.state[3, :]
            self.state[3, :] = out

        # up to right
        elif action == 10:
            self.state[0:2, :] = np.rot90(self.state[0:2, :], -1)
            out = np.copy(self.state[8, :])
            self.state[8, :] = self.state[6, :]
            self.state[6, :] = self.state[4, :]
            self.state[4, :] = self.state[2, :]
            self.state[2, :] = out

        # back to left
        elif action == 11:
            self.state[8:10, :] = np.rot90(self.state[8:10, :], -1)
            out = np.copy(self.state[6:8, 1])
            self.state[6:8, 1] = np.flip(self.state[11, :])
            self.state[11, :] = self.state[2:4, 0]
            self.state[2:4, 0] = np.flip(self.state[0, :])
            self.state[0, :] = out

        # front to left
        elif action == 12:
            self.state[4:6, :] = np.rot90(self.state[4:6, :], 1)
            out = np.copy(self.state[6:8, 0])
            self.state[6:8, 0] = np.flip(self.state[10, :])
            self.state[10, :] = self.state[2:4, 1]
            self.state[2:4, 1] = np.flip(self.state[1, :])
            self.state[1, :] = out


model = Model()
for i in range(12):
    for j in range(4):
        model.transition(action=i+1)

print(model.get_state())