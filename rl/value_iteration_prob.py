from iterative_policy_evaluation_deterministic import SMALL_ENOUGH
import numpy as np
from grid_world import ACTION_SPACE, windy_grid, windy_grid_penalized
from policy_evaluation_prob import center_string, print_values
from functools import reduce

SMALL_ENOUGH = 1e-3
GAMMA = 0.9

def print_policy(p, g):
    num = 10*g.cols+1
    print(center_string("Policy", num, '-'))
    for i in range(g.rows):
        print('|', end='')
        for j in range(g.cols):
            cell = p.get((i,j), ' ')
            print(f'{cell:^9}|', end='')
        print('')
        print('-' * num)

def get_trans_probs_and_rewards(grid):
    transition_probs = {}
    rewards = {}

    for (s, a), v in grid.probs.items():
        for s2, p in v.items():
            transition_probs[(s, a, s2)] = p
            rewards[(s, a, s2)] = grid.rewards.get(s2, 0)
    return transition_probs, rewards


if __name__ == "__main__":
    grid = windy_grid()
    # grid = windy_grid_penalized(-0.2)

    transition_probs, rewards = get_trans_probs_and_rewards(grid)

    policy = {}
    V = {}
    for s in grid.all_states():
        V[s] = 0
    while True:
        delta = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                v_old = V[s]
                values = {}
                for a in ACTION_SPACE:
                    values[a] = 0
                    for s2 in grid.all_states():
                        r = rewards.get((s, a, s2),0)
                        values[a] +=  transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
                V[s] = max(list(values.values()))
                delta  = max(delta, np.abs(v_old - V[s]))

        # print_values(V, grid)
        if delta < SMALL_ENOUGH:
            break
        print_values(V, grid)

        for s in grid.actions.keys():
            actions = {}
            for a in ACTION_SPACE:
                actions[a] = 0
                for s2 in grid.all_states():
                    r = rewards.get((s, a, s2), 0)
                    actions[a] += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
            policy[s] = list(actions.keys())[np.argmax(list(actions.values()))]

    print_values(V, grid)
    print_policy(policy, grid)



             