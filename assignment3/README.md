### Windy gridworld

The script runme.sh generates the required plots as stated in its 'usage' provided below. To obtain all the plots specified in the report, try running the executable without any arguments. The plots are stored in a folder named 'plots', upon running the script.

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
                       tuning (comparative plots of various hyperparameters)
                       best (plot the best results baseline and King models)
                       all (all of the plots in report)
  -v, --verbose        modify output verbosity

``` 

