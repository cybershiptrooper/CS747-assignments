import numpy as np

class bernoulliArms():
	def __init__(self, file):

		f = open(file)
		instances = []
		for instance in f.readlines():
			instances.append(float(instance.rstrip()))
		self.instances = np.array(instances)
		
		k = len(instances) 
		self.k = k
		self.Pavg = np.zeros(k)
		self.Psum = np.zeros(k)
		self.armpulls = np.zeros(k)
		self.totalPulls = 0;

	def print_arms(self):
		for i in self.instances:
			print(i)

	def pull(self, arm, n=1, seed = -1):
		if(seed > -1):
			np.random.seed(seed)
		rewards = np.random.binomial(1, self.instances[arm], n)
		self.updatePavg(arm, rewards)
		return rewards

	def updatePavg(self, arm, rewards):
		self.Psum[arm] += np.sum(rewards)
		self.armpulls[arm] += len(rewards)
		self.Pavg[arm] = self.Psum[arm]/self.armpulls[arm]
		self.totalPulls += len(rewards)
	
	def optimalArm(self):
		return max(self.instances)