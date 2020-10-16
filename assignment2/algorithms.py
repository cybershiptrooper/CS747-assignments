import numpy as np
import pulp

class Solver():
	"""Consists of three solving algorithms"""
	def __init__(self, arg):
		self.mdp = arg
	
	def valueIter(self):
		''' Value iteration method '''
		pass

	def policyIter(self):
		''' Howard's Policy iteration method '''
		pass

	def linearProgram(self):
		''' Linear Programming based solver '''
		pass