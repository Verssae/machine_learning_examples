from iterative_policy_evaluation_deterministic import SMALL_ENOUGH, print_policy, print_values
import numpy as np
import random
from grid_world import standard_grid, negative_grid, ACTION_SPACE
from functools import reduce
import matplotlib.pyplot as plt

CONVERGANCE = 10000
GAMMA = 0.9

def epsilon_greedy(policy, s, eps=0.1):
    if np.random.random() < eps:
        return random.choice(ACTION_SPACE)
    else:
        return policy[s]

def play(grid, policy, max_steps=20):
    # print_policy(policy, grid) 

    grid.set_state(random.choice(list(grid.actions.keys())))
    s = grid.current_state()
    a = policy[s]

    states = [s]
    actions = [a]
    rewards = [0]
    
    for _ in range(max_steps):
        r = grid.move(a)
        s = grid.current_state()

        states.append(s)
        rewards.append(r)

        if grid.game_over():
            break
        else:
            a = epsilon_greedy(policy, s)
            actions.append(a)

    return states, actions, rewards

def max_dict(dict):
    return reduce(lambda acc, cur: cur if cur[1] > acc[1] else acc, list(dict.items()), list(dict.items())[0])

if __name__ == "__main__":
    grid = standard_grid()
    Q = {}
    returns = {}
    for s in grid.actions:
        Q[s] = {}
        returns[s] = {}
        for a in ACTION_SPACE:
            Q[s][a] = 0
            returns[s][a] = []

    policy = {}
    for s in grid.actions:
        policy[s] = random.choice(ACTION_SPACE)


    print_values(grid.rewards, grid)
    print("\n")
    
    
    deltas = []
    for i in range(CONVERGANCE):
        biggest_change = 0
        states, actions, rewards = play(grid, policy)
        G = 0
        for t in range(len(rewards)-2, -1, -1):
            G = rewards[t+1] + GAMMA*G
            if (states[t] not in states[:t]) and (actions[t] not in actions[:t]):
                s, a = states[t], actions[t]
                returns[s][a].append(G)
                old_Q = Q[s][a]
                Q[s][a] = np.mean(returns[s][a])
                policy[s] = max_dict(Q[s])[0]
                biggest_change = max(biggest_change, np.abs(old_Q - Q[s][a]))
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print_values(V, grid)
    print_policy(policy, grid)

