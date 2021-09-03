import numpy as np

def prediction(policy, env, gamma=0.9):
    V = {}
    returns = {}
    for s in env.all_states():
        V[s] = 0
        returns[s] = []
    
    while True:
        states, actions, rewards = env.play(policy)
        g = 0
        for t in range(len(states), stop=0, step=-1):
            g = rewards[t+1] + gamma*g
            if states[t] not in states.values():
                returns

