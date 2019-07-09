# Report
---

## Used algorithm
This implementation uses the Deep deterministic policy gradient as explained in this [paper](https://arxiv.org/pdf/1509.02971.pdf). It is an off-policy actor-critic method which uses two seperate deep neural networks to perform both tasks. The actor-network gives the actions as real numbers, therefore this method suits perfectly for continous action spaces.

## Code structure
To start the environment and train/watch an agent, you'll just need to run the jupyter notebook `Continous_Control.ipynb`. Because previously trained model parameters are also included in this repository, you can skip the training section and directly watch an trained agent interact with the environment. 

The code is tested on the twenty-agent version of the reacher environment, although the code will also work -- without any changes -- with the single-agent version.

Firstly all the hyperparameters for the ddpg-agent are stored in an own simple class `config`, which lies in the [`Config.py`](https://github.com/ChaosMcChief/DRLND_p2_Continous_Control/blob/master/Config.py). An instance of the `config`-class has to be passed to the agent while instantiating the agent.

The [`ddpg-agent.py`](https://github.com/ChaosMcChief/DRLND_p2_Continous_Control/blob/master/ddpg_agent.py) contains the ddpg-agent alongside with the replay memory and an implementation of the Ornstein-Uhlenbeck random process to generate action-noise in order to explore the action-space. An explaination of this random process can be found on the [wikipedia-site](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)

The last python-module needed includes the neural networks for the actor and the critic in the [`model.py`](https://github.com/ChaosMcChief/DRLND_p2_Continous_Control/blob/master/model.py). The neural networks are standard feed forward networks with three layers. For the critic the state is first passed through the first layer before being concatenated with the action values.

## Hyperparameters
As mentioned above, the [`Config.py`](https://github.com/ChaosMcChief/DRLND_p2_Continous_Control/blob/master/Config.py)-file stores all the relevant hyperparameters:

```
self.BUFFER_SIZE = int(1e5)  # replay buffer size
self.BATCH_SIZE = 128        # minibatch size
self.GAMMA = 0.99            # discount factor
self.TAU = 1e-3              # for soft update of target parameters
self.LR_actor = 1e-4         # learning rate for the actor 
self.LR_critic = 1e-3        # learning rate for the critic     
self.weight_decay=0          # Weight-Decay for L2-Regularization

# Use the GPU if one is available
self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Results
The results are shown in the plot below. Interestingly the shown and best results are achieved without batchnormalization or prioritized experience replay (per). Both actual improvements result in a lack of convergence and a quick hyperparametersearch didn't change this behaviour. In the jupyter notebook you can easily enable the per and try this out. The code for the per is taken from the first project, where it achieved better results than the standard replay buffer.

![Scoreplot](https://github.com/ChaosMcChief/DRLND_p2_Continous_Control/blob/master/Scores.png)

## Ideas for future experiments
The first thing that could be tried out is taking the ddpg-approach on the next level by implementing the [D4PG](https://arxiv.org/pdf/1804.08617.pdf)-algorithm, in which the critic doesn't directly estimates the action-value-function, but estimates a probability distribution of the said function. This should improve stability in training, which -- at least in my implementation -- is kind of an issue for the ddpg-algorithm in this environment. Also the distributed training could help with stability, where multiple different agents are learning timewise seperatly and share experiences.
