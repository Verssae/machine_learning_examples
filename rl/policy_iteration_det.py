from iterative_policy_evaluation_deterministic import SMALL_ENOUGH
import numpy as np
from grid_world import ACTION_SPACE, standard_grid
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

    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]
    return transition_probs, rewards

if __name__ == "__main__":
    grid = standard_grid()

    transition_probs, rewards = get_trans_probs_and_rewards(grid)

    initial_policy = {
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
    print_policy(initial_policy, grid)
    policy = initial_policy

    V = {}
    for s in grid.all_states():
        V[s] = 0

    it = 0
    while True:
        delta = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                v_old = V[s]
                v_new = 0
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        pi = 1 if policy.get(s) == a else 0
                        r = rewards.get((s, a, s2),0)
                        v_new += pi * transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
                    V[s] = v_new
                    delta  = max(delta, np.abs(v_old - V[s]))

        print(f'iter: {it}, delta: {delta}')
        print_values(V, grid)
        if delta < SMALL_ENOUGH:
            break
            
        policy_stable = True
        for s in grid.all_states():
            if not grid.is_terminal(s):
                a_old = policy[s]
                actions = {}
                for a in ACTION_SPACE:
                    actions[a] = 0
                    for s2 in grid.all_states():
                        r = rewards.get((s, a, s2), 0)
                        actions[a] += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
                policy[s] = list(actions.keys())[np.argmax(list(actions.values()))]
                if a_old != policy[s]:
                    policy_stable = False

        print_policy(policy, grid)
        if policy_stable:
            break
    
        it += 1




             