import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from windygridworld import WindyGridWorld
from agents import Agent
import matplotlib.pyplot as plt

lr = 0.7
episodes = 200
steps = 8000 #max number of steps before termination
epsilon = 0.05
numSeeds = 50
verbose = False

def runwindy(update, king = False, stochastic = False):
	data = np.zeros(episodes)
	dataking = np.zeros(episodes)
	for i in range(numSeeds):
		windy = WindyGridWorld(king=king, stochastic = stochastic)
		numStates = windy.numStates()
		numActions = windy.numActions()
		np.random.seed(numSeeds)
		agent = Agent(numStates, numActions, update = update, lr=lr, epsilon= epsilon)
		datum = run(agent, env = windy, 
			steps = steps, episodes=episodes, verbose = False)
		data +=np.array(datum)
	return np.cumsum(data/numSeeds)

def sarsa0(king=False, stochastic=False):
	if(verbose and stochastic): 
		print("generating plot for stochastic world")
	elif(verbose and king): 
		print("generating plot for king")
	elif(verbose):
		print("generating baseline plot")
	update = "sarsa0"
	if verbose:print(update)
	x = runwindy(update, king=king, stochastic=stochastic)
	y = np.arange(episodes)
	plt.figure()
	plt.plot(x,y, 'r')
	plt.grid()
	if verbose: print(x[-1],y[-1])
	if(king): string = "king"
	else: string = "baseline"
	if(stochastic): string = "stochastic"
	plt.title("sarsa(0): "+string)
	plt.savefig("plots/"+string)

def versus_methods(king = False, stochastic = False):
	if(verbose): print("running versus_methods")
	updates = ["sarsa0","expected-sarsa","Q"]
	plt.figure()
	for update in updates:
		if(verbose): print(update)
		x = runwindy(update, king, stochastic)
		y = np.arange(episodes)
		if(verbose):print(x[-1], y[-1])
		plt.plot(x,y)
	plt.grid()
	plt.legend(['sarsa(0)', 'expected-sarsa','Q-learning'])
	figname = ""
	if(king): figname+="king_"
	else:figname+="baseline_"
	if(stochastic): figname+="stoch_"
	# string = str(numSeeds)+'seeds'+'_eps_'+str(epsilon)+'_lr_'+str(lr)
	string = "versus_methods"
	plt.title("versus_methods: "+figname[:-1])
	plt.savefig('plots/'+figname+string+'.png')


def versusWorlds():
	if(verbose): print("running versus_worlds")
	update = "sarsa0"
	if(verbose): print(update)
	x = runwindy(update)
	y = np.arange(episodes)
	plt.figure()
	plt.grid()
	plt.figure()
	if(verbose):print("base:", x[-1], y[-1])
	plt.plot(x,y)
	x = runwindy(update, king = True)
	if(verbose):print("king:", x[-1], y[-1])
	plt.plot(x,y)
	x = runwindy(update, king = True, stochastic = True)
	if(verbose):print("stochastic:", x[-1], y[-1])
	plt.plot(x,y)
	plt.legend(['base model', 'king', 'stochastic(king)'])
	# string = update+"_seeds_"+str(numSeeds)+'eps_'+str(epsilon)+'_lr_'+str(lr)
	string = "sarsa(0) for different worlds"
	plt.title(string)
	plt.savefig('plots/'+string+'.png')

def run(agent, env, steps = 2000, episodes=100,
		verbose=False):
	"""training loop :p"""
	data = []
	for e in range(episodes):
		env.start()
		x, y = env.state()
		state = int(x+10*y)
		a = agent.epsilonGreedy(state)
		for step in range(steps):
			x, y, r = env.step(a).values()
			new_state = x+10*y
			a1 = agent.epsilonGreedy(new_state)
			agent.update_Q(state, a, r, new_state, a1)
			state = new_state
			a = a1
			if(env.end()): 
				data.append(step)
				if verbose and e%(episodes//20)==0: print(step)
				break

	return data

def msg():
	return '''
./runme.sh 		 [-h] [--lr LR] [--episodes EPISODES]
                 	 [--epsilon EPSILON] [--seeds SEEDS] [--data DATA]

    OR

python generate_data.py	 [-h] [--lr LR] [--episodes EPISODES]
                 	 [--epsilon EPSILON] [--seeds SEEDS] [--data DATA]
	'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage=msg(), formatter_class= RawTextHelpFormatter)
	parser.add_argument("--lr", default = lr,
		help="learning rate of the agent"
		)
	parser.add_argument("--episodes", default = episodes, help="(integer) # episodes")
	parser.add_argument("--epsilon", default = epsilon, help= "policy's epsilon")
	parser.add_argument("--seeds", default = numSeeds,
		help="number of seeds you want to average over"
		)
	parser.add_argument("--data", default = "all", 
		help="the data you want to generate. Must be one of -\n"
			"baseline (baseline plot)\n"
			"king (plot with king's moves)\n"
			"stochastic(plot with stochastic wind and king's moves)\n"
			"versus_methods (comparative plot of update algorithms)\n"
			"versus_worlds (comparative plot of king and base model)\n"
			"all (all of the above plots)\n",
		)
	parser.add_argument("-v", "--verbose", help="modify output verbosity", 
                    action = "store_true")
	
	args = parser.parse_args()
	lr = float(args.lr)
	epsilon = float(args.epsilon)
	try:
		episodes = int(args.episodes)
	except:
		raise Exception("please enter integer episodes")
	try:
		numSeeds = int(args.seeds)
	except:
		raise Exception("please enter integer seeds")
	verbose = args.verbose
	function = args.data

	if function == "all":
		sarsa0()
		sarsa0(king=True)
		sarsa0(king=True, stochastic=True)
		versus_methods()
		versusWorlds()
	elif function == "baseline":
		sarsa0()
	elif function =="king":
		sarsa0(king=True)
	elif function =="stochastic":
		sarsa0(king=True, stochastic=True)
	elif function == "versus_worlds":
		versusWorlds()
	elif function == "versus_methods":
		versus_methods()
	else:
		raise Exception("please enter valid arguments for data")
	# versusKing(verbose=True)
	