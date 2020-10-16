import sys
import argparse
import numpy as np

class Planner():
	"""wrapper class for solving algorithms"""
	def __init__(self, mdp, algorithm):
		super(Planner, self).__init__()
		self.mdp_path = mdp
		self.algo = algorithm
		self.mdp = self.getMDP(printMDP = True)

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
		mdp = {"S":S, "A":A, "T":T, "R":R, "gamma":gamma, "start":start, "end":end, "mdptype":mdptype }

		return mdp


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	###############################################################################################
	######################## CHANGE THE DEFAULTS TO "" IN PARSER ##################################
	###############################################################################################
	parser.add_argument("--mdp", type = str, default = "data/mdp/continuing-mdp-10-5.txt", help = "Path to the mdp") 
	parser.add_argument("--algorithm", type = str, default = "vi", help=
			"Name of solving algorithm. Must be one of vi(value iteration), hpi(Howard's policy iteration), or lp(linear programming)")

	args = parser.parse_args()
	if args.mdp=="":
		raise Exception("please provide valid path for mdp")
	elif args.algorithm=="":
		raise Exception("please provide valid algorithm")
	planner = Planner(args.mdp, args.algorithm)
	#planner.printArgs()