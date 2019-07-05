import numpy as np
import random
import copy
from collections import namedtuple, deque

import models

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, config, state_size, action_size, num_agents, seed, per=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            per (bool): If True, uses prioritized experience replay
            duelling (bool): If True, uses duelling network architecture
        """

        self.config = config
        self.epsilon = self.config.EPSILON_START

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed
        self.per = per

        # Initialize the Actor and Critic Networks
        self.actor = models.Actor(state_size, action_size, seed).to(self.config.device)
        self.actor_target = models.Actor(state_size, action_size, seed).to(self.config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.config.LR_actor)

        self.critic = models.Critic(state_size, action_size, seed).to(self.config.device)
        self.critic_target = models.Critic(state_size, action_size, seed).to(self.config.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.config.LR_critic, weight_decay=self.config.weight_decay)     
        
        # Initialize the random-noise-process for action-noise
        self.is_training = True
        self.randomer = OUNoise((self.num_agents, self.action_size), self.seed)

        # Hard update the target networks to have the same parameters as the local networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Initialize the replay-buffer according to `per`
        if per:
            self.memory = PrioritizedReplayBuffer(self.config.BUFFER_SIZE, self.config.BATCH_SIZE, seed)
        else:
            self.memory = ReplayBuffer(self.config.BUFFER_SIZE, self.config.BATCH_SIZE, seed, self.config.device)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def decay_epsilon(self):
        """ Perform one decay-step for epsilon"""
        self.epsilon -= self.config.EPSILON_DECAY

    def reset(self):
        self.randomer.reset()

    def step(self, state, action, reward, next_state, done):
        """ Processes one experience-tuple (i.e store it in the replay-buffer
        and take a learning step, if it is time to do that.
        """
        
        # Save experience in replay memory
        for i in range(self.num_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.config.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.config.GAMMA)

    def act(self, states):
        """Returns actions for given state as per current policy.
        Also adds random action-noise to the action-values while training.

        Params
        ======
            state (array_like): current state  
        """       
        # Convert the state to a torch-tensor
        states = torch.from_numpy(states).float().to(self.config.device)

        # Compute the action-values
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(states)
        self.actor.train()
        action = action.cpu().numpy()
        action += self.is_training * max(self.epsilon, self.config.EPSILON_MIN) * self.randomer.noise()
        action = np.clip(action, -1.0, 1.0)

        return action

    def learn(self, mini_batch, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        # Check if per is used and Sample a minibatch 
        if self.per:
            # Get the priority-indices and the IS-weights
            (tree_idx, experiences, b_ISWeights) = mini_batch
            b_ISWeights = torch.from_numpy(b_ISWeights).to(self.config.device)
            
            # Preprocessing the experience-tuple
            states = torch.from_numpy(np.vstack([e[0][0] for e in experiences if e is not None])).float().to(self.config.device)
            actions = torch.from_numpy(np.vstack([e[0][1] for e in experiences if e is not None])).float().to(self.config.device)
            rewards = torch.from_numpy(np.vstack([e[0][2] for e in experiences if e is not None])).float().to(self.config.device)
            next_states = torch.from_numpy(np.vstack([e[0][3] for e in experiences if e is not None])).float().to(self.config.device)
            dones = torch.from_numpy(np.vstack([e[0][4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.config.device)
    
        else:
            states, actions, rewards, next_states, dones = mini_batch
            
        
        # Get the noised actions from the local network for the next states
        actions_target_next = self.actor_target(next_states)
        
        # Evaluate the computed actions with the critic-target-network
        Q_targets_next = self.critic_target(next_states, actions_target_next)

        # Compute the expected future (discounted) rewards
        y_expected = rewards + (self.config.GAMMA * Q_targets_next * (1 - dones))
        y_predicted = self.critic(states, actions)

        # Calculate the TD-Error   
        delta = y_expected-y_predicted

        # # If the memory is a PER, update the priorities and adjust the loss
        if self.per:
            # Compute the TD-Error delta
            absolute_errors = torch.abs(delta).detach().cpu().numpy()
            self.memory.batch_update(tree_idx, absolute_errors)
            # Multiply loss with the IS-Weights
            loss_critic = (delta*b_ISWeights)**2
            loss_critic = torch.mean(loss_critic)
        else:
            loss_critic = torch.mean(delta**2) # MSE-Loss

        # critic gradient
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # actor gradient
        actions_local = self.actor(states)
        loss_actor = (-self.critic(states, actions_local)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # update the target networks
        self.soft_update(self.actor, self.actor_target, self.config.TAU)                     
        self.soft_update(self.critic, self.critic_target, self.config.TAU) 


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This code is a slightly modified version of the code found in the github-repo
    from Thomas Simonini (https://github.com/simoninithomas/Deep_reinforcement_learning_Course)
    
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity, batch_size, seed):
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """

        # Initialize the Sumtree in wich the priorities and experiences are stored
        self.tree = SumTree(capacity)

        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.num_updates = 0 # Numbers of updates to determine if enough samples are in the memory for one minibatch
        
    def add(self, state, action, reward, next_state, done):
        """
        Store a new experience in our tree
        Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
        """
        
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        self.num_updates += 1
        experience = (state, action, reward, next_state, done)

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p

    def sample(self):
        """
        - First, to sample a minibatch of size k, the range [0, priority_total] is / into k ranges.
        - Then a value is uniformly sampled from each range
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element
        """

        # Create a sample array that will contain the minibatch
        memory_b = []
        n = self.batch_size
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)
        
        return (b_idx, memory_b, b_ISWeights)
    
    def batch_update(self, tree_idx, abs_errors):
        """
        Update the priorities on the tree
        """

        abs_errors += self.PER_e  # Avoid 0-error by adding the constant PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        # Loop through all priorities and aupdate the tree
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.num_updates

class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0
    
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
    

    def add(self, priority, data):
        """
        Here we add our priority score in the sumtree leaf and add the experience in data
        """
        
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """
        
        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update the leaf
        self.update (tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
            

    def update(self, tree_index, priority):
        """
        Update the leaf priority score and propagate the change through tree
        """

        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    
    def get_leaf(self, v):
        """
        Here we get the leaf_index, priority value of that leaf and experience associated with that index.
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def noise(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state