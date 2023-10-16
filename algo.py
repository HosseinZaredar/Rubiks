import time
from collections import OrderedDict
import numpy as np
import itertools
import heapq
from state import next_state, init_state


class Node:
    def __init__(self, parent, action, cost, state):
        self.parent = parent
        self.action = action
        self.cost = cost
        self.state = state


def hash_fn(state, cost=None):
    if cost is None:
        return hash(state.data.tobytes())
    else:
        state_flattened = state.flatten()
        state_action = np.zeros(state_flattened.shape[0]+1)
        state_action[:state_flattened.shape[0]] = state_flattened
        state_action[-1] = cost
        return hash(state_action.data.tobytes())
    

def get_hashed_goal_state(): # create hashed goal states
    colors = [1, 2, 3, 4, 5, 6]
    permutations = list(itertools.permutations(colors, 6))
    hashed_goal_states = set()
    for p in permutations:
        state = np.zeros((12, 2), dtype=np.uint8)
        for j in range(6):
            state[2*j:2*(j+1)] = p[j]
        hashed_goal_states.add(hash_fn(state))
    return hashed_goal_states


def backtrack(final_node):
    action_sequence = []
    current_node = final_node
    while current_node is not None:
        action_sequence.append(current_node.action)
        current_node = current_node.parent
    action_sequence = list(reversed(action_sequence))[1:]
    return action_sequence


def bfs(init_state, hashed_goal_states):

    # create dictionaries
    explored_dict = {}
    frontier_dict = OrderedDict()

    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    # creating the initial node
    initial_node = Node(None, None, 0, init_state)
    init_hashed_state = hash_fn(initial_node.state)

    # add initial node to frontier
    frontier_dict[init_hashed_state] = initial_node

    max_depth = 0

    while True:

        # checking if the stack is empty
        if len(frontier_dict) == 0:
            return None, expanded_num, explored_num

        # removing a node to explore
        hashed_state, node = frontier_dict.popitem(last=False)

        # adding to explored
        explored_dict[hashed_state] = node

        explored_num += 1  # one node explored

        # printing max depth
        if node.cost > max_depth:
            max_depth = node.cost
            print('Max Depth:', max_depth)

        # checking if it is a goal state
        if hashed_state in hashed_goal_states: 
            return node, expanded_num, explored_num

        # checking every possible move
        for i in range(1, 12+1):
            
            # create new state
            new_state = next_state(node.state, action=i)
            new_hashed_state = hash_fn(new_state)

            # checking if child's state is already explored or in frontier
            if new_hashed_state in explored_dict or new_hashed_state in frontier_dict:
                continue

            # add new node to frontier dict
            new_node = Node(node, i, node.cost + 1, new_state)
            frontier_dict[new_hashed_state] = new_node

            expanded_num += 1  # one node expanded


def heuristic(node):  # heuristic function
    # 1: h(node) = 0, =BFS
    return 0


def a_star(init_state, hashed_goal_states):

    # all expanded, is only used to prevent duplicate addition of nodes in priority queue
    all_expanded_set = set()
    
    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    # creating the initial node
    initial_node = Node(None, None, 0, init_state)
    init_hashed_state_cost = hash_fn(initial_node.state, initial_node.cost)
    all_expanded_set.add(init_hashed_state_cost)

    # add initial node to frontier
    frontier_pq = []
    heapq.heappush(frontier_pq, (heuristic(initial_node) + initial_node.cost, id(initial_node), initial_node))

    max_depth = 0

    while True:

        # checking if the priority queue is empty which means there's no solution
        if len(frontier_pq) == 0:
            return None, None, None
        
        # popping a node
        priority, node_id, node = heapq.heappop(frontier_pq)

        # creating hashed state
        hashed_state = hash_fn(node.state)

        explored_num += 1  # one node explored

        # printing max depth
        if node.cost > max_depth:
            max_depth = node.cost
            print('Max Depth:', max_depth)

        # checking if it is a goal state
        if hashed_state in hashed_goal_states: 
            return node, expanded_num, explored_num
        
        # checking every possible move
        for i in range(1, 12+1):
            
            # create new state
            new_state = next_state(node.state, action=i)
            new_hashed_state_cost = hash_fn(new_state, node.cost+1)

            # checking if the child node is not already in all_expanded_set
            if new_hashed_state_cost in all_expanded_set:
                continue

            # adding the child node to all_expanded_set
            all_expanded_set.add(new_hashed_state_cost)

            # adding the new_node to priority_queue i.e. actual frontier list
            new_node = Node(node, i, node.cost + 1, new_state)
            heapq.heappush(frontier_pq, (heuristic(new_node) + new_node.cost, id(new_node), new_node))

            expanded_num += 1  # one node expanded


def solve(init_state, method):

    if method == 'Random':
        return np.random.randint(1, 12+1, 10)
    
    elif method == 'BFS':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = bfs(init_state, hashed_goal_states)
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        return action_sequence

    elif method == 'A*':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = a_star(init_state, hashed_goal_states)        
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        return action_sequence
    
    else:
        return []