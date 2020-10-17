import sys
import argparse
import numpy as np

def encode(path):
	#get grid
	try:
		f = open(path, 'r')
	except Exception as e:
		raise Exception("please enter valid path for the maze file")
	grid = f.readlines()
	for i in range(len(grid)):
		grid[i] = grid[i].split()
	grid = np.array(grid, dtype = "int")
	f.close()

	#form mdp
	S = np.shape(grid)
	A = 4 #N, S, E, W
	start, end = np.where(grid == 2),np.where(grid == 3)
	print("numStates", np.prod(S))
	print("numActions", A)
	print("start",np.prod(start))
	print("end", end)
	#North
	for x in range(S[0]):
		for y in range(1,S[1]):
			if(grid[x,y] == 3 or grid[x,y-1]==1):
				continue
			print("transition", x*y, 0, x*(y-1),  1.0, 0.0)
	#South
	for x in range(S[0]):
		for y in range(S[1]-1):
			if(grid[x,y] == 3 or grid[x,y+1]==1):
				continue
			print("transition", x*y, 1, x*(y+1),  1.0, 0.0)
	#East
	for x in range(1,S[0]):
		for y in range(S[1]-1):
			if(grid[x,y] == 3 or grid[x-1,y]==1):
				continue
			print("transition", x*y, 2, y*(x-1),  1.0, 0.0)
	#West
	for x in range(S[0]-1):
		for y in range(S[1]):
			if(grid[x,y] == 3 or grid[x+1,y]==1):
				continue
			print("transition", x*y, 3, y*(x+1),  1.0, 0.0)

	print("mdptype", "episodic")
	print("gamma", 0.69)#what to do with gamma??

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--grid", type=str, default = "", 
		help="path to maze file that needs to be encoded")
	args = parser.parse_args()
	encode(args.grid)


#todo here:-
#what to do with gamma? what to do with reward?
#do we have multiple start and end points?
#make sure the N, S, E, W reference is correct (or does it matter)