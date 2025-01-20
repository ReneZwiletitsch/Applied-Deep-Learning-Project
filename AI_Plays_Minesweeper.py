import random
from minesweeper import MinesweeperEnv
import torch
import time
import torch.nn as nn
import torch.nn.functional as F


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



def select_action(state,done_actions,policy_net):
    with torch.no_grad():
        action_values = policy_net(state.unsqueeze(1))
        action_values[0, done_actions] = -float('inf')
        move = torch.argmax(action_values).item()
    return move


def play_a_game(policy_net):
    move_delay = 1
    done_actions=[]
    _, state = env.reset()
    state = torch.tensor(state, dtype=torch.float,device=device).unsqueeze(0)

    env.print_player_grid()
    while True:
        action = select_action(state,done_actions,policy_net)       
        observation, _, endstate,won,recursion_list,action = env.step(action,done_actions,False)
        done_actions.append(int(action))
        done_actions.extend(recursion_list)
        state = torch.tensor(observation, dtype = torch.float, device=device).unsqueeze(0)

        env.print_player_grid()
        if endstate:
            if won:
                print("VICTORY")
            else:
                print("DEFEAT")
            break

        time.sleep(move_delay)
    return won




if __name__ == '__main__':
    device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
    )
    game_size = "smallest"
    net_type = "CNN"
    xdim,ydim,total_mines = 5,5,5
    env = MinesweeperEnv(xdim,ydim,total_mines)

    policy_net = ConvNet(xdim,ydim).to(device)
    PATH = './saved_states/safed_policy_net'+net_type+game_size+'.pth'
    policy_net.load_state_dict(torch.load(PATH))



    done =False
    
    while not done:
        play_a_game(policy_net)
        done_check = input("play again? y/n\n")
        if done_check.lower() == "n":
            done = True


