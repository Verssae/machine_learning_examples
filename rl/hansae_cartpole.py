# import gym

# if __name__ == "__main__":
#     env = gym.make("CartPole-v0")

#     env.reset()
#     done = False
#     while not done:
#         a = env.action_space.sample()
#         s_next, r, done, info = env.step(a)
#         env.render(a)

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()