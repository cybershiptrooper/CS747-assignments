#!/usr/bin/local/python3
import numpy as np

class WindyGridWorld():
	"""
	Windy Gridworld environment
	arguments- 
	model:(string)
	"""
	def __init__(self, king=False, stochastic=False):
		super().__init__()
		self.king = king
		self.stochastic = stochastic
		self.W, self.H = 9, 6
		self.wind = np.array([0,0,0,1,1,1,2,2,1,0])
		self.x, self.y = 0, 3
		self.sT = [7,3]

	def start(self):
		self.x, self.y = 0, 3
	def end(self):
		return (self.x == self.sT[0]) and (self.y == self.sT[1])
	def numStates(self):
		return (self.H+1)*(self.W+1)
	def numActions(self):
		return 4 + 4*self.king


	def state(self):
		return self.x, self.y

	def step(self, action):
		'''
		args:
		action:(int) ranges from 0 to 7
		0 for UP
		1 for DOWN
		2 for RIGHT
		3 for LEFT
		4 for UP+RIGHT
		5 for DOWN+RIGHT
		6 for UP+LEFT
		7 for DOWN+LEFT
		'''
		H,W = self.H, self.W
		assert self.x<=W and self.y<=H
		assert action < 8
		assert not self.end()
		# wind
		if(self.stochastic):
			wind = np.random.randint(-1,2)
			self.y -= self.wind[self.x]*wind
		else:
			self.y += self.wind[self.x]
		# actions
		if(action in [0, 4, 6]):self.y += 1
		elif(action in [1,5,7]):self.y -= 1
		if(action in [2, 4, 5]):self.x += 1
		elif(action in [3,6,7]):self.x -= 1
		#boundary
		x,y = self.x, self.y
		self.x=0 if x<0 else (W if x>W else x)
		self.y=0 if y<0 else (H if y>H else y)
		#return reward
		return {"x": self.x, "y":self.y,"reward":-1}