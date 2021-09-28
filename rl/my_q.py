# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
from functools import reduce
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def epsilon_greedy(q, s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    return max_dict(q[s])[0]
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def max_dict(dict):
    return reduce(lambda acc, cur: cur if cur[1] > acc[1] else acc, list(dict.items()), list(dict.items())[0])

if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = negative_grid(step_cost=-0.1)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)


  Q = {}
  states = grid.all_states()
  for s in states:
    Q[s] = {}
    if grid.is_terminal(s):
      for a in ALL_POSSIBLE_ACTIONS:
        Q[s][a] = 0
    else:
      for a in ALL_POSSIBLE_ACTIONS:
        Q[s][a] = 0

  # store max change in V(s) per episode
  episode_rewards = []

  # repeat until convergence
  n_episodes = 10000
  for it in range(n_episodes):
    if it % 2000 == 0:
      print("it:", it)
    # begin a new episode
    s = grid.reset()
    
    rewards = 0
    while not grid.game_over():
      a = epsilon_greedy(Q, s)
      r = grid.move(a)
      s_next = grid.current_state()

      Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * max_dict(Q[s_next])[1] - Q[s][a])
      
      # next state becomes current state
      s = s_next
   
      rewards += r

    # store delta
    episode_rewards.append(rewards)

  plt.plot(episode_rewards)
  plt.show()
  V = {}
  for s, Qs in Q.items():
    V[s] = max_dict(Q[s])[1]
  print("values:")
  print_values(V, grid)
  Pi = {}
  for s, Qs in Q.items():
    if grid.is_terminal(s):
      Pi[s] = ''
    else:
      Pi[s] = max_dict(Q[s])[0]
  print("values:")
  print_policy(Pi, grid)

