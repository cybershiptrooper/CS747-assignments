import numpy as np
import argparse
from windygridworld import WindyGridWorld
from agents import Agent
import matplotlib.pyplot as plt

lr = 0.6
episodes = 200
steps = 10000
epsilon = 0.05
numSeeds = 50

def versus_methods(king = False, stochastic = False):
	windy = WindyGridWorld(king = king, stochastic=stochastic)
	numStates = windy.numStates()
	numActions = windy.numActions()
	updates = ["sarsa0","expected-sarsa","Q"]
	for update in updates:
		print(update)
		data = np.zeros(episodes)
		for i in range(numSeeds):
			np.random.seed(numSeeds)
			agent = Agent(numStates, numActions, update = update, lr=lr, epsilon= epsilon)
			datum = run(agent, env = windy, 
				steps = steps, episodes=episodes, verbose = False)
			data +=np.array(datum)
		data/=numSeeds
		x = np.cumsum(data)
		y = np.arange(episodes)
		print(x[-1],y[-1])
		plt.plot(x,y)
	plt.grid()
	plt.legend(['sarsa(0)', 'expected-sarsa','Q-learning'])
	figname = ""
	if(king): figname+="king_"
	if(stochastic): figname+="stoch_"
	string = 'eps_'+str(epsilon)+'_lr_'+str(lr)+'.png'
	plt.savefig('plots/'+figname+string)

def versusking():
	updates = ["sarsa0","expected-sarsa","Q"]
	for update in updates:
		data = np.zeros(episodes)
		dataking = np.zeros(episodes)
		for i in range(numSeeds):
			windy = WindyGridWorld()
			numStates = windy.numStates()
			numActions = windy.numActions()
			np.random.seed(numSeeds)
			agent = Agent(numStates, numActions, update = update, lr=lr, epsilon= epsilon)
			datum = run(agent, env = windy, 
				steps = steps, episodes=episodes, verbose = False)
			data +=np.array(datum)

			windyking = WindyGridWorld(king=True)
			numStatesking = windyking.numStates()
			numActionsking = windyking.numActions()
			np.random.seed(numSeeds)
			agent = Agent(numStates, numActions, update = update, lr=lr, epsilon= epsilon)
			datum = run(agent, env = windy, 
				steps = steps, episodes=episodes, verbose = False)
			dataking +=np.array(datum)
	

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

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# versus_methods(king=True)

	windy = WindyGridWorld(king = True)
	numStates = windy.numStates()
	numActions = windy.numActions()
	print(numActions)
	updates = ["sarsa0","expected-sarsa","Q"]
	update = updates[1]
	agent = Agent(numStates, numActions, update = update, lr=lr, epsilon= epsilon)
	datum = run(agent, env = windy, 
				steps = steps, episodes=episodes, verbose = False)
	x = np.cumsum(datum)
	y = np.arange(episodes)
	plt.plot(x,y)
	plt.savefig("plots/expected-sarsa")