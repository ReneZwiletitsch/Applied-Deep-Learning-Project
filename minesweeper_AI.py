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
import json

class ConvNet(nn.Module):
    def __init__(self,xdim,ydim):
        super().__init__()
        #out channels of first
        self.xdim = xdim
        self.ydim = ydim
        #in channels
        factor = 5
 
        first_conv_channels = 50 *factor
        second_conv_channels = 60*factor
        first_fc_out = 60
        second_fc_out = 40

        self.conv1 = nn.Conv2d(1,first_conv_channels,kernel_size=5,stride=1,padding=2)
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


class Net(nn.Module):
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
        x= torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self,*args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)


def all_the_checks():
    #checks which net to use, which game size and if we want to load
    load_check=input("load last training? (y/n)")
    CNN_check=input("use CNN? (y/n)")
    size=input("game size: \n(y) smallest \n(n) small\n")
    training_check=input("(y) train \n(n) test\n")
    game_size = "small"
    net_type = "NN"
    num_episodes=500
    if training_check.lower() == "y":
        training = True
        num_episodes=input("train for how many episodes? (0 = infinite training)")
        try:
            num_episodes = int(num_episodes)
        except:
            print("invalid number of episodes. defaulting to 500")
            num_episodes = 500

    else:
        training = False

    if size.lower() == "y": #same as in stanford paper
        game_size = "smallest"
        xdim=5
        ydim=5
        total_mines = 5
    else:
        print("defaulting to small game size")
        #game definitions
        xdim=9
        ydim=9
        total_mines = 10

    game_params = [xdim,ydim,total_mines]

    if CNN_check.lower() == "y":
        net_type = "CNN"
        policy_net = ConvNet(xdim,ydim).to(device)
        target_net = ConvNet(xdim,ydim).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        steps_done = 0
    else:
        policy_net = Net(xdim,ydim).to(device)
        target_net = Net(xdim,ydim).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        steps_done = 0

    if load_check.lower()=="y":
        print("reloading last state")
        PATH = './saved_states/safed_policy_net'+net_type+game_size+'.pth'
        policy_net.load_state_dict(torch.load(PATH))
        PATH = './saved_states/safed_target_net'+net_type+game_size+'.pth'
        target_net.load_state_dict(torch.load(PATH))
        with open('./saved_states/steps_done'+net_type+game_size, "r") as file:
            lines = file.read()
            steps_done = int(lines)
            print("loaded steps_done:",steps_done)

    training_params = [policy_net,target_net,steps_done]
    setup = [net_type,game_size]
    return game_params, training_params,setup,training,num_episodes


def optimize_model(memory):
    if len(memory) < BATCH_SIZE:
        return memory
    
    transition = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transition))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state).unsqueeze(1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    optimizer.zero_grad()

    state_action_values = policy_net(state_batch).gather(1,action_batch.to(torch.int64).unsqueeze(1))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states.unsqueeze(1)).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) +reward_batch


    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()
    return memory



def play_a_game(fast,notrandom):
    move_delay = 1
    done_actions=[]
    _, state = env.reset()
    state = torch.tensor(state, dtype=torch.float,device=device).unsqueeze(0)
    if not fast:
        env.print_player_grid()
    while True:
        action = select_action(state,done_actions,True,not_random=notrandom,steps_done=steps_done)
       # let a human play
        #temp_action=int(input("next action: "))
        #action =torch.tensor([temp_action],device=device, dtype = torch.float)
        
        observation, _, endstate,won,recursion_list,action = env.step(action,done_actions,False)
        done_actions.append(int(action))
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
    return won


def select_action(state,done_actions,testing_mode,not_random,steps_done):
    global action_amounts

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done +=1
    if (sample>eps_threshold or testing_mode) and not_random:
        with torch.no_grad():
            #move = torch.argmax(policy_net(state.unsqueeze(1)))
           # print("SELECTED FROM POLICY: ",move.item())
            action_values = policy_net(state.unsqueeze(1))
            action_values[0, done_actions] = -float('inf')
            
            move = torch.argmax(action_values).item()
            action_amounts[move] +=1
    else:
        temp_action= env.sample_action()
        while temp_action in done_actions:
            temp_action= env.sample_action()        
        move = torch.tensor([temp_action],device=device, dtype = torch.float)
    return move


def save_state(policy_net,target_net,steps_done,total_rewards,net_type,game_size):
    PATH = './saved_states/safed_policy_net'+net_type+game_size+'.pth'
    torch.save(policy_net.state_dict(), PATH)

    PATH = './saved_states/safed_target_net'+net_type+game_size+'.pth'
    torch.save(target_net.state_dict(), PATH)

    with open('./saved_states/steps_done'+net_type+game_size, "w") as file:
        file.write(str(steps_done))
    with open('./saved_states/total_rewards'+net_type+game_size+'.json', "w") as file:
        json.dump(total_rewards, file)



def training_function(env,memory,policy_net,target_net,total_rewards):
    
    _, state = env.reset()
    state = torch.tensor(state, dtype=torch.float,device=device).unsqueeze(0)
    current_total_reward = 0
    done_actions=[]

    while True:
        action = select_action(state,done_actions,False,not_random=True,steps_done=steps_done)
        observation, reward, endstate,won,recursion_list,action = env.step(action,done_actions,True)
        done_actions.append(int(action))
        done_actions.extend(recursion_list)
        current_total_reward += reward
        reward = torch.tensor([reward],device=device)
        action = torch.tensor([action],device=device)
        if endstate:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype = torch.float, device=device).unsqueeze(0)

        memory.push(state,action,next_state,reward)
        state = next_state
        memory = optimize_model(memory)

        policy_net_state_dict = policy_net.state_dict()
        target_net_state_dict = target_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if endstate:
            total_rewards.append(current_total_reward)
            break

    return env,memory,policy_net,target_net,total_rewards




# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(device)

#training hyperparameters
BATCH_SIZE = 1024
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.03
EPS_DECAY = 1000000
TAU = 0.005
#LR = 1e-4
LR = 1e-3



(xdim,ydim,total_mines), (policy_net,target_net,steps_done),(net_type,game_size),training,num_episodes = all_the_checks()

Transition = namedtuple('Transition', ('state','action','next_state','reward'))

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)


#initialize game
env = MinesweeperEnv(xdim,ydim,total_mines)
#state is the grid a player would see
_, state = env.reset()


#stats
action_amounts=[0]*81
total_rewards = []


#training loop
testing = not training
while training:
    #infinite training. doesn't need break or anything because it should be force stopped by design
    if num_episodes == 0:
        print("starting infinite training")
        i_episode = 0
        while True:
            i_episode += 1
            if i_episode %100 == 0:
                print("episode: ",i_episode)
                save_state(policy_net,target_net,steps_done,total_rewards,net_type,game_size)
            env,memory,policy_net,target_net,total_rewards = training_function(env,memory,policy_net,target_net,total_rewards)

    print("starting training")
    for i_episode in range(num_episodes):
        if i_episode %50 == 0:   
            print("action amounts")
            print(action_amounts)
            action_amounts=[0]*81
        env,memory,policy_net,target_net,total_rewards = training_function(env,memory,policy_net,target_net,total_rewards)

        if i_episode %100 == 0:
            print("episode done: ",i_episode)
            save_state(policy_net,target_net,steps_done,total_rewards,net_type,game_size)


    play_game_check=input("play game? (y/n)")
    if play_game_check=="y"or play_game_check=="Y":
        try:
          play_a_game(False,True)
        except:
            pass
    training_check=input("continue training? (y/n)")
    if training_check.lower()=="n":
        training = False
if testing:

    with open('./saved_states/total_rewards'+net_type+game_size+'.json', "r") as file:
        total_rewards = json.load(file)
    not_random = True #set to False if you want to use random moves instead of the trained model
    visual_game_amount = 5 # how many games should be showed to the user
    move_delay = 1 #how many seconds between moves for visual games
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    math.exp(-1. * steps_done / EPS_DECAY)
    reward_averages=[]
    for i in range(int(len(total_rewards)/100)):
        reward_averages.append(sum(total_rewards[i*100:(i+1)*100]))

    wins = 0
    for i in range(visual_game_amount):
        won = play_a_game(False,not_random)
        if won:
            wins += 1
        time.sleep(2)

    #extended test:
    print("Fast evaluation in progress...")
    game_amount = 1000
    wins = 0
    for i in range(game_amount):
        if i%1000 == 0:
            print("game ",i," of ",game_amount)
        won = play_a_game(True,not_random)
        if won:
            wins += 1

    print("last exploration rate: ", eps_threshold)
    print("rewards (sum of 100 episodes each): ")
    print(reward_averages)
    print("wins: ",wins, "out of: ",game_amount)
    print("win rate: ",wins/game_amount)




"""

total_rewards = total_rewards[-100:]
sum = 0
for i in total_rewards:
    sum +=i

print("average reward in 100 games: ",sum/100)

"""
