#!/usr/bin/local/python3
import numpy as np
from agents import agents

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
		self.end = [7,3]

	def end(self):
		return (self.x == 7) and (self.y == 3)
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
		assert x<=W and y<=H
		assert action < 4
		assert self.end()
		self.y += wind[x]*(y!=H)
		if(action in [0, 4, 6]):self.y += 1*(y!=H)
		elif(action in [1,5,7]):self.y -= 1*(y!=0)
		if(action in [2, 4, 6]):self.x += 0*(x!=W)
		elif(action in [3,5,7]):self.x -= 1*(x!=0)
