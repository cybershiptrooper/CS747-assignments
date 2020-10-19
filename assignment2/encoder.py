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
	start, end = np.where(grid == 2), np.prod(np.where(grid == 3), axis = 0)
	print("numStates", np.prod(S))
	print("numActions", A)
	print("start",np.prod(start))
	print("end", end=" ")
	for i in end:
		if(i == end[-1]):
			print(i)
		else:
			print(i,end=" ")
	t = 1.0
	r_norm = -0.01
	r_max = 1
	#West
	for y in range(1,S[1]-1):
		for x in range(1,S[0]-1):
			r= r_norm
			if(grid[x,y-1]==1 or grid[x,y] == 3 or grid[x,y] ==1): #On Wall/ Next to wall/ On End
				continue
			elif(grid[x,y-1] == 3): #next to end
				r = r_max
			print("transition", x+y*(S[1]), 0, x+(y-1)*S[1],  r, t)
	#East
	for y in range(1,S[1]-1):
		for x in range(1,S[0]-1):	
			r = r_norm
			if(grid[x,y] == 3 or grid[x,y+1]==1 or grid[x,y] ==1): #On Wall/ Next to wall/ On End
				continue
			elif(grid[x,y+1] == 3): #next to end
				r = r_max
			print("transition", x+y*(S[1]), 1, x+(y+1)*S[1], r, t)
	#North
	for y in range(1,S[1]-1):
		for x in range(1,S[0]-1):	
			r= r_norm
			if(grid[x,y] == 3 or grid[x-1,y]==1 or grid[x,y] ==1): #On Wall/ Next to wall/ On End
				continue
			elif(grid[x-1,y] == 3): #next to end
				r = r_max
			print("transition", x+y*(S[1]), 2, x+y*(S[1])-1,  r, t)
	#South
	for y in range(1,S[1]-1):
		for x in range(1,S[0]-1):	
			r= r_norm
			if(grid[x,y] == 3 or grid[x+1,y]==1 or grid[x,y] ==1):	#On Wall/ Next to wall/ On End
				continue
			elif(grid[x+1,y] == 3): #next to end
				r = r_max
			print("transition", x+y*(S[1]), 3, x+y*(S[1])+1,  r, t)

	print("mdptype", "episodic")
	print("gamma", 1)#what to do with gamma??

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