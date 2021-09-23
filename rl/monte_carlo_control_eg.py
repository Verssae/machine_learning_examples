from iterative_policy_evaluation_deterministic import SMALL_ENOUGH, print_policy, print_values
import numpy as np
import random
from grid_world import standard_grid, negative_grid, ACTION_SPACE
 

CONVERGANCE = 100
GAMMA = 0.9

grid = standard_grid()
Q = {}
returns = {}
for s, values in grid.actions.items():
    for a in values:
        returns[(s, a)] = []
        Q[(s, a)] = 0

print(returns)
print(Q)
policy = {}
for s, values in grid.actions.items():
    policy[s] = random.choice(values)


print_values(grid.rewards, grid)
print("\n")

def play(grid, max_steps=20):
    print_policy(policy, grid) 

    states = []
    rewards = []

    s, a = random.choice(list(Q.keys()))
    grid.set_state(s)
    rewards.append(0)

    
    r = grid.move(a)


    states.append((s, a))
    rewards.append(r)

    steps = 0
    while not grid.game_over():
        
        s = grid.current_state()
        a = policy[s]
        states.append((s, a))

        r = grid.move(a)
        rewards.append(r)

        steps += 1
        if steps >= max_steps:
            break
    # r = grid.rewards.get(s, 0)
    # rewards.append(r)
    return states, rewards

it = 0
while it < CONVERGANCE:
    states, rewards = play(grid)
    G = 0
    for t in range(len(rewards)-2, -1, -1):
        G = rewards[t+1] + GAMMA*G
        if states[t] not in states[:t]:
            s, a = states[t]
            returns[(s, a)].append(G)
            Q[(s, a)] = np.mean(returns[(s, a)])
            sas = list(filter(lambda e: e[0] == s, list(Q)))
            acts = list(map(lambda e: e[1], sas))
            policy[s] = acts[np.argmax([Q[k] for k in sas])]
    it += 1

print(Q)
print_policy(policy, grid)

