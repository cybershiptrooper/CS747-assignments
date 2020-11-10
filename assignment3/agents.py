import numpy as np

class Agent():
	"""Three solving agents-
		1. Sarsa(0)
		2. Expected Sarsa
		3. Q-Learning
		policy used: epsilon greedy
		Plus a run loop for windy gridworld
	"""
	def __init__(self, numStates, numActions, discount=1, lr = 0.5, update="sarsa0", 
			epsilon = 0.1):

		self.update_Q = self.getAgent(update)
		self.S, self.A = numStates, numActions
		self.gamma = discount
		self.epsilon = epsilon
		self.lr = lr
		self.Q = np.zeros((numStates,numActions))

	def getAgent(self, update):
		if update=="sarsa0":
			return self.sarsa0
		elif update=="expected-sarsa":
			return self.sarsaE
		elif update=="Q":
			return self.Q_Learning

	def epsilonGreedy(self,s):
		if(np.random.uniform()>self.epsilon):
			return np.argmax(self.Q[s])
		else:
			return np.random.choice(self.A)

	def sarsa0(self, s, a, r, s1, a1):
		self.Q[s,a] += + self.lr*(r + self.gamma*self.Q[s1,a1]-self.Q[s,a])

	def sarsaE(self, s, a, r, s1, a1):
		#find expected Q
		bestQ = np.max(self.Q[s1])
		expected_sample = np.sum(self.Q[s1])*self.epsilon/self.A
		expected = bestQ*(1-self.epsilon)+expected_sample
		#find target
		target = r + self.gamma*expected
		#update Q
		self.Q[s,a] += self.lr*(target-self.Q[s,a])

	def Q_Learning(self, s, a, r, s1, a1):
		self.Q[s,a] += self.lr*(r + self.gamma*np.max(self.Q[s1,a]) -self.Q[s,a])

	# def run(self, env, steps = 8000, episodes=100,
	# 		verbose=False):
	# 	data = []
	# 	for e in range(episodes):
	# 		env.start()
	# 		x, y = env.state()
	# 		state = int(x+10*y)
	# 		a = self.epsilonGreedy(state)
	# 		for step in range(steps):
	# 			x, y, r = env.step(a).values()
	# 			new_state = x+10*y
	# 			a1 = self.epsilonGreedy(new_state)
	# 			self.update_Q(state, a, r, new_state, a1)
	# 			state = new_state
	# 			a = a1
	# 			if(env.end()): 
	# 				break
	# 		data.append(step)
	# 		print(step)
	# 	return data
	