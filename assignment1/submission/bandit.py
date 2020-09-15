#usr/bin/env/python3
import sys
from algorithms import sampler
import numpy as np

class bandit(sampler):
	"""docstring for bandit"""
	def __init__(self, arg):
		super().__init__(arg[:-1])
		self.hz = int(arg[4])

	# def __init__(self, horizon, sampler):

	def run(self):
		REW = 0
		for i in range(self.hz):
			rewards = self.sample()
			REW += np.sum(rewards)

		REG = self.hz*self.optimalArm()
		return REG - REW

def main():
	if(len(sys.argv) != 6):
		print("Please enter valid arguments")
		sys.exit()
	bandit_instance = bandit(sys.argv[1:])
	print(bandit_instance.run())

if __name__ == '__main__':
	main()



#todo-
#where do I put the seeds??
#klucb, thompson
#thompson with hint
