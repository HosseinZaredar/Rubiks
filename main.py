from ursina import *
from rubik import Rubik
import numpy as np
import argparse
import time
from state import init_state, next_state
from algo import solve


if __name__ == '__main__':

    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # initializing state
    state = init_state()

    if not args.manual:

        # scramble
        scramble_sequence = np.random.randint(1, 12+1, 10)
        for a in scramble_sequence:
            state = next_state(state, action=a)

        # solve rubik
        print('SOLVING...')
        start_time = time.time()
        solve_sequence = solve(state)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'SOLVE FINISHED In {elapsed_time}S.')
        time.sleep(3)

    # start game
    app = Ursina(size=(800, 600))
    rubik = Rubik()

    if args.manual:
        input = lambda key: rubik.action(key, animation_time=0.5)
    else:
        action_dict = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
                    7: 'q', 8: 'w', 9: 'e', 10: 'r', 11: 't', 12: 'y'}

        # perform scramble + solution
        scramble_sequence = [action_dict[i] for i in scramble_sequence]
        solve_sequence = [action_dict[i] for i in solve_sequence]
        invoke(rubik.action_sequence, scramble_sequence, solve_sequence, delay=5.0)
    
    app.run()