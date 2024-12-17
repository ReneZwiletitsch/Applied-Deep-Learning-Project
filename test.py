import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple,deque
import random
from minesweeper import MinesweeperEnv
import math
from itertools import count
import json

small_field = True #True for 5x5 4mines; False for 9x9 10mines, must match training size
i_episode = 4000 #which iteration should be loaded
conv_or_not = False #True for CNN, False otherwise
not_random = True #set to False if you want to use random moves instead of the trained model
visual_game_amount = 5 # how many games should be showed to the user
move_delay = 1 #how many seconds between moves for visual games


# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Net(nn.Module):
    def __init__(self,xdim,ydim):
        super().__init__()
        #out channels of first
        self.xdim = xdim
        self.ydim = ydim
        #in channels
        first_conv_channels = 50
        second_conv_channels = 60
        first_fc_out = 60
        second_fc_out = 40

        self.conv1 = nn.Conv2d(1,first_conv_channels,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(first_conv_channels,second_conv_channels,kernel_size=5,stride=1,padding=2)
        self.fc1 = nn.Linear((self.xdim)*(self.ydim)*second_conv_channels,first_fc_out)
        #self.fc1 = nn.Linear((self.xdim)*(self.ydim),first_fc_out)
        self.fc2 = nn.Linear(first_fc_out,second_fc_out)
        self.fc3 = nn.Linear(second_fc_out,self.xdim*self.ydim)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x= torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class NonConvNet(nn.Module):
    def __init__(self,xdim,ydim):
        super().__init__()
        #out channels of first
        self.xdim = xdim
        self.ydim = ydim
        #in channels
        first_conv_channels = 10
        second_conv_channels = 60
        first_fc_out = 128
        second_fc_out = 40

        self.fc1 = nn.Linear((self.xdim)*(self.ydim),first_fc_out)
        self.fc2 = nn.Linear(first_fc_out,first_fc_out)
        self.fc3 = nn.Linear(first_fc_out,self.xdim*self.ydim)

    def forward(self,x):
        #print("#################")
        #print(x.shape)
        x= torch.flatten(x, start_dim=1)
       # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def select_action(state,done_actions,testing_mode,not_random):
    global steps_done
    global policy_counter
    global random_counter
    global action_amounts
    global total_action_amount

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done +=1


    if (sample>eps_threshold or testing_mode) and not_random:
        policy_counter += 1
        with torch.no_grad():
            #move = torch.argmax(policy_net(state.unsqueeze(1)))
           # print("SELECTED FROM POLICY: ",move.item())
            action_values = policy_net(state.unsqueeze(1))
            action_values[0, done_actions] = -float('inf')
            
            move = torch.argmax(action_values).item()
            action_amounts[move] +=1
            total_action_amount += 1
        



    else:
        random_counter +=1
        temp_action= env.sample_action()
        while temp_action in done_actions:
            temp_action= env.sample_action()        
        move = torch.tensor([temp_action],device=device, dtype = torch.float)
        

    return move




all_moves = []
def play_a_game(fast,notrandom):
    global move_delay
    done_actions=[]
    temp_moves = []
    _, state = env.reset()
    state = torch.tensor(state, dtype=torch.float,device=device).unsqueeze(0)
    if not fast:
        env.print_player_grid()
    while True:
        action = select_action(state,done_actions,True,not_random=notrandom)
        done_actions.append(int(action))
        temp_moves.append(int(action))
       # let a human play
        #temp_action=int(input("next action: "))
        #action =torch.tensor([temp_action],device=device, dtype = torch.float)
        
        observation, _, endstate,won,recursion_list = env.step(action)
        done_actions.extend(recursion_list)
        state = torch.tensor(observation, dtype = torch.float, device=device).unsqueeze(0)
        if not fast:
            env.print_player_grid()
        if endstate:
            if won:
                if not fast:
                    print("THE BOT IS USEFUL")
                #time.sleep(5)
            else:
                if not fast:
                    print("FAIL")
            break
        if not fast:
            time.sleep(move_delay)
    all_moves.append(temp_moves)
    return won



if not not_random:
    print("CAREFUL YOU ARE IN RANDOM ACTIONS MODE")
    time.sleep(3)


if small_field:
    #game definitions
    xdim=5
    ydim=5
    total_mines = 4
else:
    #game definitions
    xdim=9
    ydim=9
    total_mines = 10

#training hyperparameters
BATCH_SIZE = 1024
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.03
EPS_DECAY = 1000000
TAU = 0.005
#LR = 1e-4
LR = 1e-3


n_actions = xdim * ydim

random_counter = 0
policy_counter = 0
action_amounts=[0]*81
total_action_amount = 0


#initialize game
env = MinesweeperEnv(xdim,ydim,total_mines)


print("reloading episode: ",i_episode)
if conv_or_not:
    print("loading convolutional network")
    policy_net = Net(xdim,ydim).to(device)
    target_net = Net(xdim,ydim).to(device)
    #reload last net


    PATH = './savepointsCNN/'+str(i_episode)+'policy.pth'
    policy_net.load_state_dict(torch.load(PATH))
    PATH = './savepointsCNN/'+str(i_episode)+'target.pth'
    target_net.load_state_dict(torch.load(PATH))
    with open('./savepointsCNN/'+str(i_episode)+'steps', "r") as file:
        lines = file.read()
        steps_done = int(lines)
    with open('./savepointsCNN/'+str(i_episode)+'total_rewards.json', "r") as file:
        total_rewards = json.load(file)


else:
    print("loading non-convolutional network")
    policy_net = NonConvNet(xdim,ydim).to(device)
    target_net = NonConvNet(xdim,ydim).to(device)
    #reload last net
    PATH = './savepoints/'+str(i_episode)+'policy.pth'
    policy_net.load_state_dict(torch.load(PATH))
    PATH = './savepoints/'+str(i_episode)+'target.pth'
    target_net.load_state_dict(torch.load(PATH))
    with open('./savepoints/'+str(i_episode)+'steps', "r") as file:
        lines = file.read()
        steps_done = int(lines)
    with open('./savepoints/'+str(i_episode)+'total_rewards.json', "r") as file:
        total_rewards = json.load(file)


eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    math.exp(-1. * steps_done / EPS_DECAY)
print("last exploration rate: ", eps_threshold)



print("rewards (sum of 100 episodes each): ")
reward_averages=[]
for i in range(int(len(total_rewards)/100)):
    reward_averages.append(sum(total_rewards[i*100:(i+1)*100]))

print(reward_averages)



wins = 0
for i in range(visual_game_amount):
    won = play_a_game(False,not_random)
    if won:
        wins += 1

print(all_moves)
print("wins: ",wins, "out of: ",visual_game_amount)



print("Fast evaluation in progress...")
#extended test:
game_amount = 10000
wins = 0
for i in range(game_amount):
    if i%1000 == 0:
        print("game ",i," of ",game_amount)
    won = play_a_game(True,not_random)
    if won:
        wins += 1



print("wins: ",wins, "out of: ",game_amount)
print("win rate: ",wins/game_amount)
