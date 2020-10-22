import numpy as np
import pulp as p

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
		pi = np.zeros(self.S, dtype = "int")
		#use Bellman update iteratively
		# i = 0
		while True:
			#V(i+1) <- max(PR + PV(i), axis = actions) 
			Vmat = self.PR + self.gamma*self.T.dot(V)
			Vnew = np.max(Vmat,axis=1)
			#difference < error => done
			diff = np.sum(abs(Vnew-V))
			if (diff<= error):# or (i >= 100000):
				pi = np.argmax(Vmat,axis=1)
				break;
			V = Vnew
			# i += 1
			# if not (i%4000):
			# 	print(i, diff)
		#print(diff)
		# a = -np.prod(np.prod(T[np.arange(25)] == 0, axis=1),axis = 1)
		# pi[np.where(a == -1)] = -1
		return V, pi

	def policyIter(self, error = 1e-12):
		''' Howard's Policy iteration method '''
		#initialise Value function and policy
		pi, V = np.zeros(self.S, dtype = "int"), np.zeros(self.S)
		#iterate over policy
		i = 0
		while True:
			#evaluate current policy
			j = 0
			while True:
				Vmat = self.PR + self.gamma*self.T.dot(V)
				Vpi = Vmat[np.arange(self.S), pi]
				#difference < error => done
				diff = np.sum(abs(Vpi-V))
				if diff <= error:# or j >= 1000:
					break
				V = Vpi
				# j+=1
				# if(not j%100):
				# 	print(j, diff)
			#improve all improvable states greedily
			newpi = np.argmax(Vmat, axis = 1)
			if((newpi == pi).all()): #or i >= 1000:
				break;
			pi = newpi
			i+=1
			if(not i%1000):
				print(i, diff)
		return V, pi

	def linearProgram(self, error = 1e-12):
		''' Linear Programming based solver '''
		#Create LP Minimization problem
		lp_problem = p.LpProblem('best-Vpi', p.LpMinimize)

		#create problem variables
		V = []
		for i in range(self.S):
			V.append(p.LpVariable("v"+str(i), 
				cat = p.LpContinuous))
		#objective function
		lp_problem += p.lpSum(V)
		#constraints
		PV = (self.T).dot(V)
		constraints = self.PR + self.gamma*PV
		for s in range(self.S):
			for a in range(self.A):
				lp_problem += V[s] >= constraints[s][a]

		#print(lp_problem)
		status = lp_problem.solve(p.PULP_CBC_CMD(msg = 0)) #solve
		#print(p.LpStatus[status])   # The solution status 

		Vstar = np.zeros(self.S)
		for i in range(self.S):
			Vstar[i] = p.value(V[i])

		#use the Bellman backup to get back pistar
		PVstar = (self.T).dot(Vstar)
		pistar = np.argmax(self.PR + self.gamma*PVstar, axis = 1)
		#do policy evaluation for finding V*(lp gives inaccurate V*)
		i = 0
		Vstar = np.zeros(self.S)
		while True:
			Vmat = self.PR + self.gamma*self.T.dot(Vstar)
			Vpi = Vmat[np.arange(self.S), pistar]
			#difference < error => done
			diff = np.sum(abs(Vpi-Vstar))
			if diff <= error:# or j >= 1000:
				break
			Vstar = Vpi

		return Vstar, pistar