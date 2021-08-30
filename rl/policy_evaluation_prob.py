from iterative_policy_evaluation_deterministic import SMALL_ENOUGH
from numpy.random import gamma
from grid_world import ACTION_SPACE, windy_grid
from functools import reduce
import numpy as np

SMALL_ENOUGH = 1e-3

def center_string(s, num, blank=' '):
    front = (num - len(s)) // 2
    return blank * front + s + blank * (num-front-len(s))

def print_policy(p, g):
    num = 10*g.cols+1
    print(center_string("Policy", num, '-'))
    for i in range(g.rows):
        print('|', end='')
        for j in range(g.cols):
            cell = p.get((i,j), {' ': 0})
            print(f'{reduce(lambda x, y: x + " / " + y, cell.keys()):^9}|', end='')
        print('')
        print('|', end='')
        for j in range(g.cols):
            cell = p.get((i,j), {' ': 0})
            print(f'{reduce(lambda x, y: str(x)+" "+str(y), cell.values()):^9}|', end='')
        print('')
        print('-' * num)

def print_values(V, g):
    num = 7*g.cols+1
    print(center_string("Values", num, '-'))
    for i in range(g.rows):
        print('|', end='')
        for j in range(g.cols):
            v = V.get((i,j), 0)
            if v >= 0:
                print(f' {v:.2f} |', end='')
            else:
                print(f'{v:.2f} |', end='')
        print('')
        print('-' * num)

if __name__ == "__main__":
    grid = windy_grid()

    policy = {
        (2, 0): {'U': 0.5, 'R': 0.5},
        (1, 0): {'U': 1.0},
        (0, 0): {'R': 1.0},
        (0, 1): {'R': 1.0},
        (0, 2): {'R': 1.0},
        (1, 2): {'U': 1.0},
        (2, 1): {'R': 1.0},
        (2, 2): {'U': 1.0},
        (2, 3): {'L': 1.0},
    }
    print_policy(policy, grid)

    V = {}
    for s in grid.all_states():
        V[s] = 0
    
    gamma = 0.9

    it = 0
    while True:
        delta = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in ACTION_SPACE:
                    pi = policy[s].get(a, 0)
                    trans_dict = grid.probs.get((s, a))
                    for s2 in trans_dict:
                        trans_prob = trans_dict.get(s2)
                        r = grid.rewards.get(s2, 0)
                        new_v += pi * trans_prob * (r + gamma * V[s2])
                V[s] = new_v
                delta = max(delta, np.abs(old_v - V[s]))
        print(f"> iter: {it}, delta: {delta:.4f}")
        print_values(V, grid)
        it += 1

        if delta < SMALL_ENOUGH:
            break


                

                    