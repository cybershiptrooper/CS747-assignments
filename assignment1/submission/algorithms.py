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
	global argmax, kl, isclose
	def argmax(mat):
		optimal_arms = np.where(mat==np.max(mat))[0]
		argmax = np.random.choice(len(optimal_arms))
		arm = optimal_arms[argmax]
		return arm

	def kl(p,q):
		if(p == 0):
			return (1-p)*np.log((1-p)/(1-q))
		if p==1:
			return p*np.log(p/q)

		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

	def isclose(a, b, precision=1e-06):
		return abs(a-b) <= precision #&& (b>a)

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

	def klUCB(self, c = 3, precision = 1e-06):
		#round robin
		reward = self.roundRobin()
		if(not (reward is None)): return reward

		klucb = np.zeros(self.k)
		t = self.totalPulls
		RHS = np.log(t) + c*np.log(np.log(t))
		#make klucb matrix
		for i in range(self.k):
			p = self.Pavg[i]
			#boundary
			if(p == 1 or RHS < 0):
				klucb[i] = p
				continue
			#binary search
			lb, ub = p, 1.0
			q = (ub + p)/2.0
			LHS = kl(p,q)
			#loop until within precision
			while(not isclose(LHS , RHS, precision)):
				if(LHS > RHS): ub = q
				elif(LHS < RHS):lb = q
				q = (ub + lb)/2.0
				LHS = kl(p,q)
			#print(lb, ub, LHS, RHS, q,'-------', sep = '\n')
			#update klucb
			klucb[i] = q

		#get arm to pull
		np.random.seed(self.seed)
		arm = argmax(klucb)

		#return seeded reward
		return self.pull(arm, seed = self.seed)

	def thompson(self):
		#create beta choice vector
		s = self.Psum; #Sum of rewards = number of success for bernoulli
		f = self.armpulls - s
		np.random.seed(self.seed)
		beta = np.random.beta(s+1, f+1)
		#choose maximal beta as arm
		np.random.seed(self.seed)
		arm = argmax(beta)
		return self.pull(arm, seed = self.seed)

	def hintedThompson(self):
		pass