### Windy gridworld

The script runme.sh generates comparative plots as stated in its 'usage' provided below-

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
  --data DATA          the data you want to generate: must be one of 
                       1. versus_methods (comparative plot of update algorithms)
                       2. versusKing (comparative plot of king and base model)
                       3. all (all of the above plots)
  -v, --verbose        modify output verbosity
```
