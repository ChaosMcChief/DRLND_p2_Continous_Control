import numpy as np
import random
import copy
from collections import namedtuple, deque

import model

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, config, state_size, action_size, num_agents, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.config = config

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)

        # Initialize the Actor and Critic Networks
        self.actor = model.Actor(state_size, action_size, seed).to(self.config.device)
        self.actor_target = model.Actor(state_size, action_size, seed).to(self.config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.config.LR_actor)

        self.critic = model.Critic(state_size, action_size, seed).to(self.config.device)
        self.critic_target = model.Critic(state_size, action_size, seed).to(self.config.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.config.LR_critic, weight_decay=self.config.weight_decay)     
        
        # Initialize the random-noise-process for action-noise
        self.is_training = True
        self.randomer = OUNoise((self.num_agents, self.action_size), seed)

        # Hard update the target networks to have the same parameters as the local networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Initialize replay-buffer
        self.memory = ReplayBuffer(self.config.BUFFER_SIZE, self.config.BATCH_SIZE, seed, self.config.device)


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
        if self.is_training:
            action += self.randomer.noise()
        action = np.clip(action, -1.0, 1.0)

        return action

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """


        states, actions, rewards, next_states, dones = experiences
            
        
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
        loss_critic = F.mse_loss(y_expected, y_predicted)

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


class OUNoise:
    """Ornstein-Uhlenbeck process as in https://github.com/doctorcorral/DRLND-p2-continuous/blob/master/ddpg_agent.py"""

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