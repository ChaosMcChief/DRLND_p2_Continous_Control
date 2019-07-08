# DRLND_p2_Continous_Control
This repository is a possible solution to the second Project of the Deep reinforcement learning Nanodegree - the continous-control task.

## Intro
The environment in this project is derived from the Unity Reacher environment and the solution is written for Pytorch. A quick description to the environment as well as to the state- and action-space can be found on the [Udacity Deep reinforcement learning github repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control). The solution for this project uses the code from the first project as boilerplate code and reuses elements such as the replay memory and other helperfunctions.

## What's in here?
This solution uses the DDPG-approach as described in this [paper](https://arxiv.org/pdf/1509.02971.pdf). The jupyter notebook sets up the environment, trains an agent and let you watch the trained agent.

## Instructions
This repo is tested in a Windows 64bit OS. If you use any different operating system, you have to set up the environment accordingly as describe in the already mentioned [udacity repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control) under "Getting Started". And don't forget to change the path to the environment folder in the Notebook.

The module `model.py` contains the definitions of the Pytorch-models and `ddpg_agent.py` contains the definition of the agent as well as the experience memory.
In the jupyter-notebook `Continous_Control.ipynb` lies everything you need to start the environment and train the agent.

## ToDos
This solution doesn't contain useful improvements e.g. prioritized experience replay (PER). I experimented a bit with PER, but although the agent learned faster than without PER and reaches scores of >35 fairly quick, the scores deteriorated eventually and got stuck around 22. Trying out different random seeds didn't change this behaviour. So there's a bit bugfixing to do. 
