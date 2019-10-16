from itertools import count
import math
import random
import argparse

import torch
import torch.optim as optim

from GameInteraction import GameInteraction
from MyModel import MyModel
from ExperienceReplay import ReplayMemory
from LossComputing import ComputeLoss

### compute option  : number of games to play ###
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nb_games", type=int,
                    help="Number of games to play", default = 2000)

args = parser.parse_args()

num_episodes = args.nb_games
print(" ")
print("Number of games to play : " + str(num_episodes))
#################################################

#
env_id = "CartPole-v0"
GI = GameInteraction(env_id)

### Parameters ###
LEARNING_RATE = 0.0075

epsilon_start = 1
epsilon_final = 0.0001
epsilon_decay = 5000
# Decaying epsilon function
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

MEMORY_SIZE = 1000 # nb of experiences to store and train on 

GAMMA = 0.99 # cf MDP

HIDDEN_LSTM_SIZE = 64
##################

# generate screen to get shapes
init_screen = GI.get_screen()
_, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = GI.env.action_space.n

# Initiate model
policy_net = MyModel(screen_height, screen_width, HIDDEN_LSTM_SIZE, n_actions)
target_net = MyModel(screen_height, screen_width, HIDDEN_LSTM_SIZE, n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

Memory = ReplayMemory(MEMORY_SIZE)

# create loss computer class
CL = ComputeLoss(HIDDEN_LSTM_SIZE, GAMMA, screen_height, screen_width)

eps = 0
for i_episode in range(num_episodes):
    # Initialize the environment and state
    GI.reset()
    last_screen = GI.get_screen()
    current_screen = GI.get_screen()
    state = current_screen - last_screen
    
    # Empty experiences
    States = []
    Actions = []
    Next_states = []
    Rewards = []
    Dones = []
    
    # Initiate lstm hidden layer
    hidden = (torch.zeros(HIDDEN_LSTM_SIZE).unsqueeze(0).unsqueeze(0), 
             torch.zeros(HIDDEN_LSTM_SIZE).unsqueeze(0).unsqueeze(0))
    
    for t in count():
        
        # Select an action
        eps += 1
        epsilon = epsilon_by_frame(eps)
        action, hidden = policy_net.act(state, hidden, epsilon)
        
        # Perform it and get environnement response
        reward, done = GI.act(action)

        # Observe new state
        last_screen = current_screen
        current_screen = GI.get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition
        States.append(state)
        Actions.append(action)
        Next_states.append(next_state)
        Rewards.append(reward)
        Dones.append(done)

        # Move to the next state
        state = next_state
        
        if done :
            break
    
    # Store experiences
    experience = States, Actions, Next_states, Rewards, Dones
    Memory.push(experience)
    
    # Perform one step of the optimization on this experience
    loss = CL.compute(experience, policy_net, target_net)
    
    # Perform one step of the optimization on a random experience
    experience = Memory.sample()[0]
    loss = CL.compute(experience, policy_net, target_net)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("    [Session " + str(i_episode+1) + "/" + str(num_episodes) + "] Duration : " + str(t+1))
    # Update the target network, copying all weights and biases in DQN
    if random.random() < 0.3:
        target_net.load_state_dict(policy_net.state_dict())

GI.env.close()

