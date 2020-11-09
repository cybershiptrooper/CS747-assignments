import numpy as np

class Agent():
	"""Three solving agents-
		1. Sarsa(0)
		2. Expected Sarsa
		3. Q-Learning
		Plus a run loop
	"""
	def __init__(self, numStates, numActions, end=None, reward=None, discount=1, agent="sarsa", seed=None, eps=100):
		self.agent = self.getAgent(agent)
		self.S, self.A = numStates, numActions
		self.end = end
		self.reward = reward
		self.discount = discount
		self.eps = eps
		if(seed is not None): np.random.seed(seed)
		self.Q = np.zeros(S,A)

	def getAgent(self, agent):
		if agent=="sarsa0":
			return self.sarsa0
		elif agent=="expected-sarsa"
			return self.sarsaE
		elif agent=="Q":
			return self.Q_Learning

	def run(self, verbose=False):
		pass

	def sarsa0(self):
		pass

	def sarsaE(self):
		pass

	def Q_Learning(self):
		pass
	