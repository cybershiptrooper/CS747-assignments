import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from windygridworld import WindyGridWorld
from agents import Agent
import matplotlib.pyplot as plt

lr = 0.6
episodes = 200
steps = 8000
epsilon = 0.1
numSeeds = 50


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

def versus_methods(king = False, stochastic = False, verbose=False):
	windy = WindyGridWorld(king = king, stochastic=stochastic)
	numStates = windy.numStates()
	numActions = windy.numActions()
	updates = ["sarsa0","expected-sarsa","Q"]
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
	if(stochastic): figname+="stoch_"
	string = str(numSeeds)+'seeds'+'_eps_'+str(epsilon)+'_lr_'+str(lr)+'.png'
	plt.savefig('plots/'+figname+string)


def versusKing(verbose = False):
	updates = ["sarsa0","expected-sarsa","Q"]
	for update in updates:
		if(verbose): print(update)
		x = runwindy(update)
		y = np.arange(episodes)
		plt.figure()
		plt.grid()
		if(verbose):print("base:", x[-1], y[-1])
		plt.plot(x,y)
		x = runwindy(update, king = True)
		if(verbose):print("king:", x[-1], y[-1])
		plt.plot(x,y)
		plt.legend(['base model', 'king'])
		string = update+str(numSeeds)+'eps_'+str(epsilon)+'_lr_'+str(lr)+'.png'
		plt.savefig('plots/versusking_'+string)

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
	parser.add_argument("--lr", default = 0.6,
		help="learning rate of the agent"
		)
	parser.add_argument("--episodes", default = 200, help="(integer) # episodes")
	parser.add_argument("--epsilon", default = 0.1, help= "policy's epsilon")
	parser.add_argument("--seeds", default = 50,
		help="number of seeds you want to average over"
		)
	parser.add_argument("--data", default = "all", 
		help="the data you want to generate: must be one of \n"+
			"1. versus_methods (comparative plot of update algorithms)\n"+
			"2. versusKing (comparative plot of king and base model)\n"+
			"3. all (all of the above plots)\n",
			
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
		versus_methods(verbose=verbose)
		# versus_methods(king=True, verbose=verbose)
		# versus_methods(king=True, stochastic = True, verbose=verbose)
		# versus_methods(stochastic=True, verbose=verbose)
		# versusKing(verbose=verbose)
	elif function == "versusKing":
		versusKing(verbose=verbose)
	elif function == "versus_methods":
		versus_methods()
		versus_methods(king=True, verbose=verbose)
		versus_methods(king=True, stochastic = True, verbose=verbose)
		versus_methods(stochastic=True, verbose=verbose)
	else:
		raise Exception("please enter valid arguments for data")
	# versusKing(verbose=True)
	