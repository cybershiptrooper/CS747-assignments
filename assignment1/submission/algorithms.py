import numpy as np
from arms import *

class sampler(bernoulliArms):
	"""Various Bandit Sampling algorithms"""
	def __init__(self, arg):
		super().__init__(arg[0])
		self.algo = arg[1]
		self.seed = int(arg[2])
		self.eps = float(arg[3])

	def sample(self):
		'''choose algo'''
		if(self.algo == "epsilon-greedy"):return self.epsilonGreedy()
		if(self.algo == "ucb"):return self.ucb()
		if(self.algo == "kl-ucb"):return self.klUCB()
		if(self.algo == "thompson-sampling"):return self.thompson()
		return self.hintedThompson()

	#utils
	global argmax
	def argmax(mat):
		optimal_arms = np.where(mat==np.max(mat))[0]
		argmax = np.random.choice(len(optimal_arms))
		arm = optimal_arms[argmax]
		return arm

	#algos
	def roundRobin(self):
		'''Pull each arm one time'''
		for arm in range(self.k):
			if(self.armpulls[arm] == 0):
				return self.pull(arm, seed = self.seed)
		return None

	def epsilonGreedy(self):
		np.random.seed(self.seed)
		s = np.random.uniform()
		if(s < self.eps):
			#choose random arm
			np.random.seed(self.seed)
			arm = np.random.choice(self.k)
		else:
			#choose a random arm with max Pavg
			np.random.seed(self.seed)
			arm = argmax(self.Pavg)

		#return seeded reward
		return self.pull(arm, seed = self.seed)
			
	def ucb(self):
		#do round robin if nobody sampled
		reward = self.roundRobin()
		if(not (reward is None)):
			return reward

		#calculata uta, ucb
		pulls = self.armpulls * 1.0
		uta = np.ones_like(pulls)
		uta[:] *= ( ((2 * np.log(self.totalPulls))) / pulls[:] )**0.5
		ucb = self.Pavg + uta

		#sample max ucb
		np.random.seed(self.seed)
		arm = argmax(ucb)

		#return seeded reward
		return self.pull(arm, seed = self.seed)

	def klUCB(self):
		pass

	def thompson(self):
		pass

	def hintedThompson(self):
		pass
