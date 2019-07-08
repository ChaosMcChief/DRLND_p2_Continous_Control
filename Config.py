# Defines all the hyperparams in a config-class
import torch

class config():
    
    def __init__(self):
        ### Define the hyperparameters for the agent
        self.BUFFER_SIZE = int(1e5)  # replay buffer size
        self.BATCH_SIZE = 128        # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 1e-3              # for soft update of target parameters
        self.LR_actor = 1e-4         # learning rate for the actor 
        self.LR_critic = 1e-3        # learning rate for the critic     
        self.weight_decay=0          # Weight-Decay for L2-Regularization

        # Use the GPU if one is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
