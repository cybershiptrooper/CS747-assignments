import numpy as np
import argparse

def readfile(path):
	try:
		f = open(path, 'r')
	except Exception as e:
		raise Exception(path+" is not a valid path.")
	grid = f.readlines()
	for i in range(len(grid)):
		grid[i] = grid[i].split()
	f.close()
	return grid

def mazestep(action):
	dirs = ['W','E','N','S']
	try:
		print(dirs[int(action)], end=' ')
	except Exception as e:
		raise Exception("Invalid value_policy file entered.")

	if(action == 0):
		return [0, -1]
	if(action == 1):
		return [0,1]
	if(action == 2):
		return [-1, 0]
	if(action == 3):
		return [1, 0]
	

def runMaze(grid, policy):
	x, y = np.where(grid == 2)
	i = 0
	while grid[x,y] != 3 and i < np.prod(grid.shape):
		action = policy[x+y*grid.shape[1]]
		step = mazestep(action)
		x += step[0]
		y += step[1]
		i+=1
	#print(i)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--grid", type=str, default = "", 
		help="path to maze file that needs to be encoded")
	parser.add_argument("--value_policy", type=str, default = "", 
		help="path to the file containing value function and policy og the grid")
	args = parser.parse_args()

	grid = readfile(args.grid)
	grid = np.array(grid, dtype = "int")
	
	policy = readfile(args.value_policy)
	policy = np.array(np.array(policy)[:,1], dtype = int)
	
	runMaze(grid, policy)