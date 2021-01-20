# Agent-Environment Cycle (AEC) Games

This repo uses the Reinforcement Library toolkit `RLlib` from [Ray](https://github.com/ray-project/ray). This specific wheel of the `Ray` package is included, to install it use `pip install -U ray-wheel.whl`. Install the other required packages using the command `pip install -r requirements.txt`. Required Python version is `3.7.6`.

## Learning the games

* Run `python train_pursuit.py ENV METHOD` to train on ENV with the RL method METHOD.

	* `ENV` is either `pruned` for pursuit with reward pruning or `unpruned` for pursuit without reward pruning.

	* `METHOD` should be `ADQN` for Apex DQN or `PPO` for PPO. 

## Plots and Results

The result files and plots are located under `plots/`. Use `python plot_results.py adqn` or `python plot_resuls.py ppo` to generate the result plots for Apex DQN and PPO, respectively.  
