#! /usr/bin/python
import argparse
import numpy as np
from algorithms import Solver

class Planner(Solver):
	"""wrapper class for solving algorithms"""
	def __init__(self, mdp, algorithm):
		self.mdp_path = mdp
		self.algorithm = algorithm
		mdp = self.getMDP(printMDP = False)
		super().__init__(mdp["S"], mdp["A"], mdp["T"], 
			mdp["R"], mdp["gamma"])

	def printArgs(self):
		print(self.mdp_path, self.algo)

	def getMDP(self, printMDP = False):
		f = open(self.mdp_path, 'r')
		#get number of states
		S = int(f.readline().split()[-1])
		A = int(f.readline().split()[-1])
		#get start, end
		start = int(f.readline().split()[-1])
		end = np.array(f.readline().split()[1:], dtype = "int")
		#get transition probability measure and reward function
		T = np.zeros((S,A,S), dtype="float64") 
		R = np.zeros_like(T)
		line = []
		for i in range(S*A*S):
			line = f.readline().split()
			if(line[0]!="transition"):
				break
			s1, ac, s2,  = int(line[1]), int(line[2]), int(line[3])
			R[s1, ac, s2] = float(line[-2])
			T[s1, ac, s2] = float(line[-1])
		#get discount
		mdptype = line[-1]
		gamma = float(f.readline().split()[-1])

		#print if required
		if(printMDP):
			print("type:",mdptype)
			print("number of states:", S)
			print("number of actions:", A)
			print("start:", start)
			print("end states:", end)
			print("discount:", gamma)
			#print("","","",sep = "\n")
			print("T shape:", T.shape)
			#print("","","",sep = "\n")
			print("R shape:",R.shape)

		f.close()
		mdp = {"S":S, "A":A, "T":T, "R":R, "gamma":gamma, 
		"start":start, "end":end, "mdptype":mdptype }

		return mdp

	def solve(self, error = 1e-12):
		if(self.algorithm=="vi"):
			return self.valueIter(error=error)
		elif(self.algorithm=="lp"):
			return self.linearProgram(error=error)
		elif(self.algorithm=="hpi"):
			return self.policyIter(error=error)
		else:
			raise Exception("please enter valid solver algorithm")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--mdp", type = str, default = "", help = "Path to the mdp") 
	parser.add_argument("--algorithm", type = str, default = "", help="Name of solving algorithm."+
		" Must be one of vi(value iteration), hpi(Howard's policy iteration), or lp(linear programming)")

	args = parser.parse_args()
	if args.mdp=="":
		raise Exception("please provide valid path for mdp")
	elif args.algorithm=="":
		raise Exception("please provide valid solver algorithm")
	planner = Planner(args.mdp, args.algorithm)
	V, pi= planner.solve()
	for i in range(len(V)):
		print(V[i], pi[i])