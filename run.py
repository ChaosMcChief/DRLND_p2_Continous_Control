import os
from collections import deque
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from tqdm import tqdm

import Config
from ddpg_agent_orig import Agent

def setup_env():
    env = UnityEnvironment(file_name='Reacher_Windows_x86_64_twenty/Reacher.exe')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    return env, brain, brain_name

def prepop_memory(agent, env, action_size):
    
    print("Prepopulating Memory...")
    pretrain_length = agent.memory.tree.capacity 
    
    actions = 2*np.random.rand(pretrain_length//agent.num_agents, agent.num_agents, action_size) - 1
    
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations

    for i in tqdm(range(actions.shape[0])):
               
        # Random action
        action = actions[i]
        
        # Take the action
        env_info = env.step(action)[brain_name] 

        # Get next_state, reward and done
        next_state = env_info.vector_observations   # get the next state
        reward = env_info.rewards                   # get the reward
        done = env_info.local_done                  # see if episode has finished

        # Store the experience in the memory
        # Save experience in replay memory
        for i in range(agent.num_agents):
            agent.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        #agent.memory.add(state, action, reward, next_state, done)
               
        # Reset env if done
        if done:
            env_info = env.reset(train_mode=True)[brain_name] # reset the environment
            state = env_info.vector_observations
        else:
            state = next_state
    pickle.dump(agent.memory,open('agent_memory_per_init.pkl','wb'))
    return agent

def train_agent(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, model_suff=""):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations
        score = np.zeros(agent.num_agents)

        # Decay the action-noise and reset randomer
        agent.decay_epsilon()
        agent.reset()

        while True:
            
            action = agent.act(state)
            env_info = env.step(action)[brain_name]         # send the action to the environment
            next_state = env_info.vector_observations       # get the next state
            reward = env_info.rewards                       # get the reward
            done = env_info.local_done                      # see if episode has finished
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += np.array(reward)
           
            if np.any(done): 
                break 

        scores_window.append(np.mean(score))       # save most recent score
        scores.append(np.mean(score))              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}\tLast Score: {:.2f}'.format(i_episode, np.mean(scores_window), scores[-1]), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.critic.state_dict(), f"checkpoint_critic_episode_{model_suff}.pth")
            torch.save(agent.actor.state_dict(), f"checkpoint_actor_episode_{model_suff}.pth")
            pickle.dump(scores, open(f"Scores_interrim_{model_suff}.pkl", 'wb'))
        if (np.array(scores_window)>=30).all() or (i_episode>=n_episodes+1):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.critic.state_dict(), f"checkpoint_critic_episode_final_{model_suff}.pth")
            torch.save(agent.actor.state_dict(), f"checkpoint_actor_episode_final_{model_suff}.pth")
            pickle.dump(scores, open(f"Scores_interrim_{model_suff}.pkl", 'wb'))
            break
    return scores

train=True

# Setup the environment
env, brain, brain_name = setup_env()
env_info = env.reset(train_mode=True)[brain_name]

# Define the config and kind of Replaybuffer
config = Config.config()
per = True

# number of actions
action_size = brain.vector_action_space_size

# dimension of the state-space 
num_agents = env_info.vector_observations.shape[0]
state_size = env_info.vector_observations.shape[1]

# Create the Agent and let it train
agent = Agent(config=config, 
                state_size=state_size, 
                action_size=action_size, 
                num_agents=num_agents, 
                seed=0, per=per)
# agent = Agents(state_size=state_size, 
#                 action_size=action_size, 
#                 num_agents=num_agents, 
#                 random_seed=0)
if per:
#     if os.path.exists('agent_memory_per_init.pkl'):
#         agent.memory = pickle.load(open('agent_memory_per_init.pkl','rb'))
#     else:
    prepop_memory(agent, env, action_size)

scores = train_agent(n_episodes=500, model_suff='wper')
pickle.dump(scores, open('Scores_per.pkl', 'wb'))
env.close()


#os.system("shutdown /s /t 1")
# fig = plt.figure()

# ax1 = plt.plot(range(len(scores_w_per)), scores_w_per)
# ax2 = plt.plot(range(len(scores_wo_per)), scores_wo_per)

# fig.legend(ax1, ax2, 'Mit PER', 'Ohne PER')
