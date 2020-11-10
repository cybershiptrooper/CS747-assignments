### Windy gridworld

The script runme.sh generates comparative plots as stated in its 'usage' provided below. To obtain all the plots specified in the report, try running the executable without any arguments.

```
usage: 
./runme.sh 		 [-h] [--lr LR] [--episodes EPISODES]
                 	 [--epsilon EPSILON] [--seeds SEEDS] [--data DATA]


optional arguments:
  -h, --help           show this help message and exit
  --lr LR              learning rate of the agent
  --episodes EPISODES  (integer) # episodes
  --epsilon EPSILON    policy's epsilon
  --seeds SEEDS        number of seeds you want to average over
  --data DATA          the data you want to generate. Must be one of -
                       baseline (baseline plot)
                       king (plot with king's moves)
                       stochastic(plot with stochastic wind and king's moves)
                       versus_methods (comparative plot of update algorithms)
                       versus_worlds (comparative plot of king and base model)
                       all (all of the above plots)
  -v, --verbose        modify output verbosity
``` 

The default values of lr, epsilon, episodes and numSeeds are 0.7, 0.05, 200 and 50 respectively. They were used to generate plots in the report.
