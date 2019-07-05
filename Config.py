# Defines all the hyperparams in a config-class
import torch

class config():
    
    def __init__(self):
        ### Define the hyperparameters for the agent
        self.BUFFER_SIZE = int(1e6)  # replay buffer size
        self.BATCH_SIZE = 512        # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 1e-3              # for soft update of target parameters
        self.LR_actor = 1e-4         # learning rate for the actor 
        self.LR_critic = 1e-4        # learning rate for the critic
        self.EPSILON_START = 1       # Start-value of epsilon for action-noise
        self.EPSILON_MIN = 0.001     # Min-value of epsilon
        self.EPSILON_DECAY = 0.002   # Stepsize of the decay of epsilon
               
        self.weight_decay=0

        # Use the GPU if one is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
