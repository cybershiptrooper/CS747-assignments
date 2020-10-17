import numpy as np
import pulp

class Solver():
	"""Consists of three solving algorithms"""
	def __init__(self, S, A, T, R, gamma ):
		self.S = S 			#numStates
		self.A = A 			#numActions
		self.T = T 			#SxAxS
		self.R = R			#SxAxS
		self.gamma = gamma  #[0,1]

		#other util variables
		self.PR = (T*R).sum(axis=2)

	def valueIter(self, error = 1e-12):
		''' Value iteration method '''
		#init V0 as 0
		V = np.zeros(self.S)
		pi = np.zeros_like(V)
		#use Bellman update iteratively
		# i = 0
		while True:
			#V(i+1) <- max(PR + PV(i), axis = actions) 
			Vmat = self.PR + self.gamma*self.T.dot(V)
			Vnew = np.max(Vmat,axis=1)
			#difference < error => done
			diff = np.sum(abs(Vnew-V))
			if (diff<= error): #or (i >= 100000):
				pi = np.argmax(Vmat,axis=1)
				break;
			V = Vnew
			# i += 1
			# if not (i%1000):
			# 	print(i, diff)
		#print(diff)
		return V, pi

	def policyIter(self, error = 1e-12):
		''' Howard's Policy iteration method '''
		#initialise Value function and policy
		pi, V = np.zeros(self.S, dtype = "int"), np.zeros(self.S)
		#iterate over policy
		i = 0
		while True:
			#evaluate current policy
			#j = 0
			while True:
				Vmat = self.PR + self.gamma*self.T.dot(V)
				Vpi = Vmat[np.arange(self.S), pi]
				#difference < error => done
				diff = np.sum(abs(Vpi-V))
				if diff <= error:# or j >= 1000:
					break
				V = Vpi
			#improve all improvable states greedily
			newpi = np.argmax(Vmat, axis = 1)
			if((newpi == pi).all()): #or i >= 1000:
				break;
			pi = newpi
			#i+=1
		#print(i)
		return V, pi

	def linearProgram(self, error = 1e-12):
		''' Linear Programming based solver '''
		pass