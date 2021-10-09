
import numpy as np
import matplotlib.pyplot as plt
from approx_prediction import ALL_POSSIBLE_ACTIONS, ALPHA, GAMMA, epsilon_greedy
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_policy, print_values
from sklearn.kernel_approximation import Nystroem, RBFSampler

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'R', 'L')
ACTION2INT = {a: i for i, a in enumerate(ALL_POSSIBLE_ACTIONS)}
INT2ONEHOT = np.eye(len(ALL_POSSIBLE_ACTIONS))

def epsilon_greedy(greedy, s, eps=0.1):
	p = np.random.random()
	if p < (1 - eps):
		return greedy[s]
	else:
		return np.random.choice(ALL_POSSIBLE_ACTIONS)

def one_hot_encoding(action):
	# v = np.zeros(len(ALL_POSSIBLE_ACTIONS))
	# i = ALL_POSSIBLE_ACTIONS.index(action)
	# v[i] = 1
	# return v
	return INT2ONEHOT[ACTION2INT[action]]

def psi(s, a):
	return np.concatenate((s, one_hot_encoding(a)))

def gather_samples(grid, n_episodes=10000):
	samples = []
	for _ in range(n_episodes):
		s = grid.reset()
		while not grid.game_over():
			a = np.random.choice(ALL_POSSIBLE_ACTIONS)
			samples.append(psi(s, a))
		
			r = grid.move(a)
			s = grid.current_state()
	return samples

class Model:
	def __init__(self, grid):
		# fit the featurizer to data
		samples = gather_samples(grid)

		# self.featurizer = Nystroem()
		self.featurizer = RBFSampler()
		self.featurizer.fit(samples)
		dims = self.featurizer.n_components

		# initialize linear model weights
		self.w = np.zeros(dims)

	def predict(self, s, a):
		x = self.featurizer.transform([psi(s, a)])[0]
		return x @ self.w

	def grad(self, s, a):
		x = self.featurizer.transform([psi(s, a)])[0]
		return x
	
	def max_q(self, s, args=False):
		if args:
			return ALL_POSSIBLE_ACTIONS[np.argmax(list(map(lambda action: self.predict(s, action), ALL_POSSIBLE_ACTIONS)))]
		return np.max(list(map(lambda action: self.predict(s, action), ALL_POSSIBLE_ACTIONS)))


if __name__ == '__main__':
	# grid = standard_grid()
	grid = negative_grid(step_cost=-0.1)
	print('rewards:')
	print_values(grid.rewards, grid)

	greedy_policy = {
		(2, 0): 'U',
		(1, 0): 'U',
		(0, 0): 'R',
		(0, 1): 'R',
		(0, 2): 'R',
		(1, 2): 'R',
		(2, 1): 'R',
		(2, 2): 'R',
		(2, 3): 'U',
	}

	model = Model(grid)
	mse_per_episode = []

	n_episodes = 20000
	for it in range(n_episodes):
		if (it + 1) % 100 == 0:
			print(it + 1)
	
		s = grid.reset()

		n_steps = 0
		episode_err = 0
		while not grid.game_over():
			a = epsilon_greedy(greedy_policy, s)
			r = grid.move(a)

			s2 = grid.current_state()

			if grid.is_terminal(s2):
				target = r
			else:
				target = r + GAMMA * model.max_q(s2)
			
			g = model.grad(s, a)
			err = target - model.predict(s, a)
			model.w += ALPHA * err * g

			n_steps += 1
			episode_err += err * err

			s = s2

		mse = episode_err / n_steps
		mse_per_episode.append(mse)
	
	plt.plot(mse_per_episode)
	plt.title("MSE per episode")
	plt.show()

	V = {}
	states = grid.all_states()
	for s in states:
		if s in grid.actions:
			V[s] = model.max_q(s)
		else:
			V[s] = 0
	
	print('values:')
	print_values(V, grid)

	V = {}
	states = grid.all_states()
	for s in states:
		if s in grid.actions:
			V[s] = model.max_q(s, args=True)
		else:
			V[s] = ' '
	
	print('policy:')
	print_policy(V, grid)


