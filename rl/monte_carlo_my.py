from iterative_policy_evaluation_deterministic import SMALL_ENOUGH, print_policy, print_values
import numpy as np
import random
from grid_world import standard_grid, negative_grid
 

CONVERGANCE = 50
GAMMA = 0.9

def play(grid, policy, max_steps=20):
    states = []
    rewards = []

    grid.set_state(random.choice(list(grid.actions.keys())))
    s = grid.current_state()
    states.append(s)
    rewards.append(0)

    steps = 0
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        rewards.append(r)
        s = grid.current_state()
        states.append(s)

        steps += 1
        if steps >= max_steps:
            break
    # r = grid.rewards.get(s, 0)
    # rewards.append(r)
    return states, rewards


if __name__ == "__main__":
    # grid = standard_grid()
    grid = standard_grid()
    V = {}
    returns = {}
    for s in grid.all_states():
        V[s] = 0
        returns[s] = []
    
    # policy = {}
    # for s in grid.actions.keys():
    #     policy[s] = np.random.choice(ACTION_SPACE)
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L',
    }

    print_policy(policy, grid) 
    print_values(grid.rewards, grid)
    print("\n")
    
    it = 0
    while it < CONVERGANCE:
        biggest_change = 0
        states, rewards = play(grid, policy)

        # print(states)
        # print(rewards)

        G = 0
        for t in range(len(rewards)-2, -1, -1):
            G = rewards[t+1] + GAMMA*G
            if states[t] not in states[:t]:
                s = states[t]
                returns[s].append(G)
                V[s] = np.mean(returns[s])
        it += 1

    print_values(V, grid)
