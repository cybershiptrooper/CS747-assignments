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

def runwindy(update, king = False, stochastic = False, rate=None, e=None):
	data = np.zeros(episodes)
	dataking = np.zeros(episodes)
	for i in range(numSeeds):
		np.random.seed(i)
		windy = WindyGridWorld(king=king, stochastic = stochastic)
		numStates = windy.numStates()
		numActions = windy.numActions()
		lrate = rate if rate is not None else lr
		eps = e if e is not None else epsilon
		agent = Agent(numStates, numActions, update = update, lr=lrate, epsilon= eps)
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

def best():
	if(verbose):print("plotting best method for baseline")
	x = runwindy('Q', king =False, stochastic=False, rate = 1.1, e = 0.0)
	y = np.arange(episodes)
	print(x[-1],y[-1])
	plt.figure()
	plt.plot(x,y, 'r')
	plt.title("Best method: Q-learning on baseline")
	plt.grid()
	plt.annotate("cumulative steps="+(str(x[-1])[:-2]),(x[-1]-1500,y[-1]) )
	plt.savefig("plots/best_baseline")

	if(verbose):print("plotting best method for king")
	x = runwindy('Q', king =True, stochastic=False, rate = 1.2, e = 0.0)
	y = np.arange(episodes)
	print(x[-1],y[-1])
	plt.figure()
	plt.plot(x,y, 'r')
	plt.title("Best method: Q-learning on king's move")
	plt.grid()
	plt.annotate("cumulative steps="+(str(x[-1])[:-2]),(x[-1]-800,y[-1]) )
	plt.savefig("plots/best_king")

def versusWorlds():
	if(verbose): print("running versus_worlds")
	update = "sarsa0"
	if(verbose): print(update)
	x = runwindy(update)
	y = np.arange(episodes)
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
	string = "sarsa(0)-on-different-worlds"
	plt.title(string)
	plt.grid()
	plt.savefig('plots/'+string+'.png')

def versus_hyper():
	update = "Q"
	epsilons = [0.05, 0.01, 0.001, 0.0]
	lrs = [ 0.7, 0.9 , 1.1, 1.3]

	if(verbose): print("comparing epsilons")
	plt.figure()
	for e in epsilons:
		x = runwindy(update, e=e)
		y = np.arange(episodes)
		print("epsilon:",e, x[-1],y[-1])
		plt.plot(x,y)
	string = "Q-learning-on-different-epsilons"
	plt.legend(['epsilon=0.05','epsilon=0.01','epsilon=1e-3','epsilon=0.0'])
	plt.title(string)
	plt.grid()
	plt.savefig('plots/'+string+'.png')

	if(verbose): print("comparing learning rates")
	plt.figure()
	for rate in lrs:
		x = runwindy(update, rate=rate)
		y = np.arange(episodes)
		print("rate:",rate, x[-1],y[-1])
		plt.plot(x,y)
	string = "Q-learning-on-different-lr"
	plt.legend(['lr = 0.7','lr = 0.9','lr = 1.1','lr = 1.3'])
	plt.title(string)
	plt.grid()
	plt.savefig('plots/'+string+'.png')


def run(agent, env, steps = 2000, episodes=100,
		verbose=False):
	'''training loop :p'''
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
				break
		data.append(step)
		if verbose and e%(episodes//20)==0: print(step)
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
			"tuning (comparative plots of various hyperparameters)\n"
			"best (plot the best results baseline and King models)\n"
			"all (all of the plots in report)\n",
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
		best()
		versus_hyper()
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
	elif function == "tuning":
		versus_hyper()
	elif function == "best":
		best()
	else:
		raise Exception("please enter valid arguments for data")
	# versusKing(verbose=True)
	