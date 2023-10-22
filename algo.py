from collections import OrderedDict
import numpy as np
import heapq
from state import next_state, solved_state
from location import next_location


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


class Node:
    def __init__(self, parent, action, cost, state, location=None):
        self.parent = parent
        self.action = action
        self.cost = cost
        self.state = state
        if location is not None:
            self.location = location


def hash_fn(state, cost=None):
    if cost is None:
        return hash(state.data.tobytes())
    else:
        state_flattened = state.flatten()
        state_action = np.zeros(state_flattened.shape[0]+1)
        state_action[:state_flattened.shape[0]] = state_flattened
        state_action[-1] = cost
        return hash(state_action.data.tobytes())
    

def get_hashed_goal_state():
    hashed_goal_states = {}
    state = solved_state()
    hashed_goal_states[hash_fn(state)] = state
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

        # checking if the queue is empty
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


def dls(init_state, hashed_goal_states, limit):

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
        hashed_state, node = frontier_dict.popitem(last=True)

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

        # checking if it the children would exceed the limit
        if node.cost == limit:
            continue

        # checking every possible move
        for i in range(1, 12+1):
            
            # create new state
            new_state = next_state(node.state, action=i)
            new_hashed_state = hash_fn(new_state)
            new_node = Node(node, i, node.cost + 1, new_state)

            # checking if child's state is already explored
            if new_hashed_state in explored_dict:
                existing_node = explored_dict[new_hashed_state]
                if existing_node.cost <= new_node.cost:
                    continue

            # checking if child's state is already expanded
            if new_hashed_state in frontier_dict:
                existing_node = frontier_dict[new_hashed_state]
                if existing_node.cost <= new_node.cost:
                    continue

            frontier_dict[new_hashed_state] = new_node
            expanded_num += 1  # one node expanded


def ids(init_state, hashed_goal_states, max_limit):
    for depth in range(1, max_limit+1):
        final_node, expanded_num, explored_num = dls(init_state, hashed_goal_states, limit=depth)
        if final_node is not None:
            return final_node, expanded_num, explored_num
    return None, expanded_num, explored_num


def heuristic(location):  # heuristic function
    
    # return 0  # h(node) = 0, =BFS

    select_idx = location.flatten() - 1
    output_array = np.choose(select_idx, matrix)
    h = np.sum(output_array) / 4
    return h


def a_star(init_state, init_location, hashed_goal_states):

    # all expanded, is only used to prevent duplicate addition of nodes in priority queue
    all_expanded_set = set()
    
    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    # creating the initial node
    initial_node = Node(None, None, 0, init_state, init_location)
    init_hashed_state_cost = hash_fn(initial_node.state, initial_node.cost)
    all_expanded_set.add(init_hashed_state_cost)

    # add initial node to frontier
    frontier_pq = []
    heapq.heappush(frontier_pq, (heuristic(initial_node.location) + initial_node.cost, id(initial_node), initial_node))

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
            new_location = next_location(node.location, action=i)
            new_hashed_state_cost = hash_fn(new_state, node.cost+1)

            if node.cost+1 > 14:
                continue

            # checking if the child node is not already in all_expanded_set
            if new_hashed_state_cost in all_expanded_set:
                continue

            # adding the child node to all_expanded_set
            all_expanded_set.add(new_hashed_state_cost)

            # adding the new_node to priority_queue i.e. actual frontier list
            new_node = Node(node, i, node.cost + 1, new_state, new_location)
            heapq.heappush(frontier_pq, (heuristic(new_node.location) + new_node.cost, id(new_node), new_node))

            expanded_num += 1  # one node expanded


def bi_backtrack(s_final_node, g_final_node):

    # creating the path
    action_sequence = []

    # forward
    current = s_final_node
    while current.parent is not None:
        action_sequence.insert(0, current.action)
        current = current.parent

    # backward
    current = g_final_node
    reverse_dict = {1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12,
                    7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6}
    while current.parent is not None:
        action_sequence.append(reverse_dict[current.action])
        current = current.parent

    return action_sequence


def is_found(g_frontier_dict, s_frontier_dict):  # checks if there's any common state in given frontier dictionaries

    # checking if s_frontier_dict and g_frontier_dict have anything in common
    hashed_common_state = None
    for i in s_frontier_dict:
        if i in g_frontier_dict:
            hashed_common_state = i
            break

    # if any state if found
    if hashed_common_state is not None:
        g_final_node = g_frontier_dict[hashed_common_state]
        s_final_node = s_frontier_dict[hashed_common_state]
        return s_final_node, g_final_node
    else:
        return None, None


def one_step_bfs(frontier_dict, explored_set, depth):  # exploring nodes with in the given depth

    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    while True:

        # checking if the queue is empty
        if len(frontier_dict) == 0:
            return
        
        # getting the node at the front of the queue
        hashed_state = next(iter(frontier_dict))
        node = frontier_dict[hashed_state]

        # if its depth is bigger than what is supposed to be explored, we're done
        if node.cost != depth:
            return expanded_num, explored_num
        
        explored_num += 1
        
        # adding node's state to explored_set
        explored_set.add(hashed_state)

        # popping the node's state from frontier_dict
        frontier_dict.pop(hashed_state)

        # checking every possible move
        for i in range(1, 12+1):

            # create new state
            new_state = next_state(node.state, action=i)
            new_hashed_state = hash_fn(new_state)

            # checking if child's state is already explored or in frontier
            if new_hashed_state in explored_set or new_hashed_state in frontier_dict:
                continue

            # add new node to frontier dict
            new_node = Node(node, i, node.cost + 1, new_state)
            frontier_dict[new_hashed_state] = new_node

            expanded_num += 1  # one node expanded


def bibfs(init_state, hashed_goal_states):
    
    explored_num = 0  # total number of nodes explored
    expanded_num = 0  # total number of nodes expanded

    # explored sets
    s_explored_set = set()
    g_explored_set = set()

    # frontier ordered dictionaries
    s_frontier_dict = OrderedDict()
    g_frontier_dict = OrderedDict()

    # creating the initial node
    initial_node = Node(None, None, 0, init_state)
    init_hashed_state = hash_fn(initial_node.state)

    # add initial node to frontier
    s_frontier_dict[init_hashed_state] = initial_node

    # add goal nodes to frontier
    for g in hashed_goal_states.keys():
        goal_state = hashed_goal_states[g]
        goal_node = Node(None, None, 0, goal_state)
        g_frontier_dict[g] = goal_node

    # checking if solution is already found
    s_final_node, g_final_node = is_found(g_frontier_dict, s_frontier_dict)
    if s_final_node is not None:
        return s_final_node, g_final_node, expanded_num, explored_num

    depth = 0
    while True:

        print('Max Depth:', depth+1)

        # checking if any of queues are empty which means there's no solution
        if len(s_frontier_dict) == 0 or len(g_frontier_dict) == 0:
            return None, None, expanded_num, explored_num
        
        # exploring one depth backward
        expanded_num_added, explored_num_added = one_step_bfs(s_frontier_dict, s_explored_set, depth)
        expanded_num += expanded_num_added
        explored_num += explored_num_added

        # checking if solution is found
        s_final_node, g_final_node = is_found(g_frontier_dict, s_frontier_dict)
        if s_final_node is not None:
            return s_final_node, g_final_node, expanded_num, explored_num

        # exploring one depth forward
        expanded_num_added, explored_num_added = one_step_bfs(g_frontier_dict, g_explored_set, depth)
        expanded_num += expanded_num_added
        explored_num += explored_num_added

        # checking if solution is found
        s_final_node, g_final_node = is_found(g_frontier_dict, s_frontier_dict)
        if s_final_node is not None:
            return s_final_node, g_final_node, expanded_num, explored_num

        depth += 1


def solve(init_state, init_location, method):

    if method == 'Random':
        return np.random.randint(1, 12+1, 10)
    
    elif method == 'BFS':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = bfs(init_state, hashed_goal_states)
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence

    elif method == 'A*':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = a_star(
            init_state, init_location, hashed_goal_states)        
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence

    elif method == 'DLS':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = dls(init_state, hashed_goal_states, limit=7)        
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence
    
    elif method == 'IDS':
        hashed_goal_states = get_hashed_goal_state()
        final_node, expanded_num, explored_num = ids(init_state, hashed_goal_states, max_limit=9)        
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = backtrack(final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence
    
    elif method == 'BiBFS':
        hashed_goal_states = get_hashed_goal_state()
        s_final_node, g_final_node, expanded_num, explored_num = bibfs(init_state, hashed_goal_states)        
        print('#Expanded', expanded_num)
        print('#Explored', explored_num)
        action_sequence = bi_backtrack(s_final_node, g_final_node)
        print('#Actions', len(action_sequence))
        if len(action_sequence) == 0:
            print('Failed!')
        else:
            print('Success!')
        return action_sequence
    
    else:
        return []