import time
from collections import OrderedDict
import numpy as np
import itertools
from state import next_state



class Node:
    def __init__(self, parent, action, cost, state):
        self.parent = parent
        self.action = action
        self.cost = cost
        self.state = state


def BFS(init_state):

    # create hashed goal states
    colors = [1, 2, 3, 4, 5, 6]
    permutations = list(itertools.permutations(colors, 6))
    hashed_goal_states = set()
    for p in permutations:
        state = np.zeros((12, 2), dtype=np.uint8)
        for j in range(6):
            state[2*j:2*(j+1)] = p[j]
        hashed_goal_states.add(hash(state.data.tobytes()))

    # create dictionaries
    explored_dict = {}
    frontier_dict = OrderedDict()

    # creating the initial node
    initial_node = Node(None, None, 0, init_state)
    init_hashed_state = hash(initial_node.state.data.tobytes())

    # add initial node to frontier
    frontier_dict[init_hashed_state] = initial_node

    while True:

        # checking if the stack is empty
        if len(frontier_dict) == 0:
            return 'FAIL', None

        # removing a node to explore
        hashed_state, node = frontier_dict.popitem(last=False)

        # adding to explored
        explored_dict[hashed_state] = node

        # checking if it is a goal state
        if hashed_state in hashed_goal_states: 
            return 'SUCCESS!', node
        
        else:

            # checking every possible move
            for i in range(1, 12+1):
                
                # create new state and node
                new_state = next_state(node.state, action=i)
                new_node = Node(node, i, node.cost + 1, new_state)
                new_hashed_state = hash(new_state.data.tobytes())

                # checking if child's state is already explored
                if new_hashed_state in explored_dict:
                    existing_node = explored_dict[new_hashed_state]
                    if existing_node.cost <= new_node.cost:
                        continue

                # checking if child's state is already in frontier
                if new_hashed_state in frontier_dict:
                    existing_node = frontier_dict[new_hashed_state]
                    if existing_node.cost <= new_node.cost:
                        continue

                # add to frontier dict
                frontier_dict[new_hashed_state] = new_node


def solve(initial_state):

    status, node = BFS(initial_state)
    action_sequence = []

    current_node = node
    while current_node is not None:
        action_sequence.append(current_node.action)
        current_node = current_node.parent

    action_sequence = list(reversed(action_sequence))[1:]
    print(status)

    return action_sequence