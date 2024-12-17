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


Transition = namedtuple('Transition', ('state','action','next_state','reward'))


class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self,*args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)


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

#initialize game
env = MinesweeperEnv(xdim,ydim,total_mines)

#state = playergrid
_, state = env.reset()

n_observations = len(state)

policy_net = Net(xdim,ydim).to(device)
target_net = Net(xdim,ydim).to(device)

#policy_net = NonConvNet(xdim,ydim).to(device)
#target_net = NonConvNet(xdim,ydim).to(device)
target_net.load_state_dict(policy_net.state_dict())
steps_done = 0

#reload last net?
load_check="n"
load_check=input("load last training? (y/n)")
if load_check=="y"or load_check=="Y":
    print("reloading last state")
    PATH = './safed_policy_net.pth'
    policy_net.load_state_dict(torch.load(PATH))
    PATH = './safed_target_net.pth'
    target_net.load_state_dict(torch.load(PATH))
    with open('./steps_done', "r") as file:
        lines = file.read()
        steps_done = int(lines)
        print("loaded steps_done:",steps_done)



optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)


action_amounts=[0]*81
total_action_amount = 0
def select_action(state,done_actions,testing_mode):
    global steps_done
    global policy_counter
    global random_counter
    global action_amounts
    global total_action_amount

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done +=1
    if steps_done%1000 == 0:
        print(steps_done, " steps done")
        print(eps_threshold)


    if (sample>eps_threshold or testing_mode):
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

total_rewards = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transition = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transition))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state).unsqueeze(1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1,action_batch.to(torch.int64).unsqueeze(0))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states.unsqueeze(1)).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) +reward_batch


    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(0))

    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()



def plot_results(total_rewards):
    plt.plot(total_rewards)


def play_a_game():
    done_actions=[]
    _, state = env.reset()
    state = torch.tensor(state, dtype=torch.float,device=device).unsqueeze(0)
    env.print_player_grid()
    while True:
        action = select_action(state,done_actions,True)
        done_actions.append(int(action))
       # let a human play
        #temp_action=int(input("next action: "))
        #action =torch.tensor([temp_action],device=device, dtype = torch.float)
        
        observation, _, endstate,won,recursion_list = env.step(action)
        done_actions.extend(recursion_list)
        state = torch.tensor(observation, dtype = torch.float, device=device).unsqueeze(0)

        env.print_player_grid()
        if endstate:
            if won:
                print("THE BOT IS USEFUL")
            else:
                print("FAIL")
            break
        time.sleep(0.5)



#training loop
num_episodes = 500


# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


random_counter = 0
policy_counter = 0
training = True
while training:
    for i_episode in range(num_episodes):
        if i_episode%10 ==0 and i_episode!=0:
            plot_results(total_rewards)

        _, state = env.reset()
        state = torch.tensor(state, dtype=torch.float,device=device).unsqueeze(0)
        current_total_reward = 0
        done_actions=[]
        t = -1
        if i_episode %10 == 0:
            print("episode: ",i_episode)
        if i_episode %50 == 0:   
            print("action amounts")
            print(action_amounts)
            print(total_action_amount)
            action_amounts=[0]*81
            total_action_amount = 0
        #if policy_counter>0:
            #print(random_counter,policy_counter)
        random_counter = 0
        policy_counter = 0

        while True:
            t+=1
            #if t%100 == 0:
            #   print(random_counter,policy_counter)
            #  print(t)
            action = select_action(state,done_actions,False)
            done_actions.append(int(action))
            observation, reward, endstate,won,recursion_list = env.step(action)
            done_actions.extend(recursion_list)
            current_total_reward += reward
            reward = torch.tensor([reward],device=device)
            action = torch.tensor([action],device=device)
            if endstate:
                if won:
                    print("IT'S INSANE")
                    time.sleep(1)
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype = torch.float, device=device).unsqueeze(0)

            memory.push(state,action,next_state,reward)
            state = next_state
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if endstate:
                total_rewards.append(current_total_reward)
                break


    play_game_check=input("play game? (y/n)")
    if play_game_check=="y"or play_game_check=="Y":
        try:
          play_a_game()
        except:
            pass
    training_check=input("continue training? (y/n)")
    if training_check=="n"or training_check=="N":
        training = False


#print(total_rewards)


PATH = './safed_policy_net.pth'
torch.save(policy_net.state_dict(), PATH)

PATH = './safed_target_net.pth'
torch.save(target_net.state_dict(), PATH)

with open('./steps_done', "w") as file:
    file.write(str(steps_done))

plot_results(total_rewards)

total_rewards = total_rewards[-100:]
sum = 0
for i in total_rewards:
    sum +=i

print(sum/100)


