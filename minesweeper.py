import time
import random
import copy




class MinesweeperEnv():
    """
    creates a minesweeper env

    bombs have index -2
    unopened spaces have index -1
    """
    def __init__(self,xdim,ydim,total_mines):
        self.xdim = xdim
        self.ydim = ydim
        self.total_mines = total_mines
        self.bomb_penalty = -1 #-(xdim*ydim*2)
        self.open_reward = 0.1 #1
        self.win_reward = 1 #xdim*ydim
        self.repetition_penalty = -0.5
        self.adjacent_reward = 0.1
        self.game_grid, self.player_grid = self.create_grids(self.xdim,self.ydim,self.total_mines)
        self.minecount = self.total_mines
        self.safe_spots = self.xdim*self.ydim -self.total_mines
        self.valid_actions =  [(x, y) for x in range(xdim + 1) for y in range(ydim + 1)]

    def create_grids(self,xdim,ydim,total_mines):
        #create empty grids
        row = [-1]*xdim
        game_grid= []
        for i in range(ydim):
            game_grid.append(row.copy()) 
        player_grid = copy.deepcopy(game_grid)
        #place mines
        minecount = 0
        while minecount<total_mines:
            x= random.randint(0,xdim-1)
            y = random.randint(0,ydim-1)
            if game_grid[y][x] != -2:
                game_grid[y][x] = -2
                minecount+=1

        #calculate numbers
        for y in range(len(game_grid)):
            for x in range(len(game_grid[0])):
                if game_grid[y][x] != -2:
                    current_count = 0
                    for i in range(-1,2):
                        for j in range(-1,2):
                            if x+j < xdim and y+i < ydim and x+j>=0 and y+i >= 0:
                                if game_grid[y+i][x+j] == -2:
                                    current_count +=1
                    game_grid[y][x] = current_count
        return game_grid,player_grid

    def open(self,x,y,recursion_list=[]):
        reward = 0
        endstate = False
        #check if new field is opened
        if self.player_grid [y][x] == -1:
            recursion_list.append([x,y])
            self.player_grid[y][x] = self.game_grid[y][x]

            #check if bomb
            if self.game_grid[y][x] == -2:
                endstate = True
                reward += self.bomb_penalty
                if self.minecount+self.safe_spots >= self.xdim*self.ydim:
                    reward = 0
                   # print("SAFED FROM FAIL AT FIRST OPEN")
                    self.safe_spots += 1
                    endstate = False
                    self.game_grid, self.player_grid = self.create_grids(self.xdim,self.ydim,self.total_mines)
                    reward,endstate,recursion_list=self.open(x,y,recursion_list)

            elif (self.game_grid[y][x] == 0):
                for i in range(-1,2):
                    for j in range(-1,2):
                        if  x+j < self.xdim and y+i < self.ydim and x+j>=0 and y+i >= 0 and not (i ==0 and j == 0) and not [x+j,y+i] in recursion_list:
                            tempreward,endstate,recursion_list=self.open(x+j,y+i,recursion_list)
                            reward += tempreward

            elif len(recursion_list)==1:
                #print("didn't hit 0")
                adjacent_counter = 0
                for i in range(-1,2):
                    for j in range(-1,2):
                        if  x+j < self.xdim and y+i < self.ydim and x+j>=0 and y+i >= 0 and not (i ==0 and j == 0):
                            if self.player_grid[y+i][x+j] != -1:
                                adjacent_counter += 1
                            else:
                                break
                reward += self.adjacent_reward * adjacent_counter
            
            if not endstate:
                self.safe_spots -= 1
                reward += self.open_reward
                if self.safe_spots == 0:
                    reward+=self.win_reward  
                    endstate = True    

        else:
            reward = self.repetition_penalty

        return reward,endstate,recursion_list
    

    def step(self, action,done_actions,training_mode):
        """
        Assume action is the index of the neuron i guess
        
        """
        curr_x = int(action / self.ydim)
        curr_y = int(action % self.ydim)
        fail_states = []

        HER_chance = 0.7 #chance that HER avoids hitting a mine


        temp_player_grid = copy.deepcopy(self.player_grid)

        her_passed = False
        while not her_passed:
            reward,endstate,recursion_list = self.open(curr_x,curr_y,[])
            actionlist = [x * self.ydim + y for x, y in recursion_list]

            won = False
            if endstate:
                if self.safe_spots == 0:
                    won = True

            #Hindsight Experience Replay implementation
            # if the game is over but not won, we reset the player grid to before the action and choose a different action
            if training_mode and endstate and not won:
                fail_states.append(action)
                if random.random()< HER_chance:
                    self.player_grid = copy.deepcopy(temp_player_grid) #reset player grid
                    action = self.sample_action() #choose different action at random
                    while action in done_actions or action in fail_states: #failstates makes sure to not pick the same bomb twice in one turn
                        action = self.sample_action()     
                    curr_x = int(action / self.ydim)
                    curr_y = int(action % self.ydim)
                else:
                    her_passed = True
            else:
                her_passed = True

        return self.player_grid,reward,endstate,won,actionlist,action


    def print_player_grid(self):
        underscores = " "
        underscores += "_ " * len(self.player_grid[0])
        print(self.minecount, self.safe_spots)
        print(underscores)
        for i in self.player_grid:
            row = "|"
            for j in i:
                if j>= 0:
                    row += (str(j)+"|")
                elif j==-1:
                    row += (" "+"|")
                else:
                    row += ("X"+"|")
            print(row)
            print(underscores)

    def reset(self):
        self.game_grid, self.player_grid = self.create_grids(self.xdim,self.ydim,self.total_mines)
        self.minecount = self.total_mines
        self.safe_spots = self.xdim*self.ydim -self.total_mines
        
        
        return self.game_grid,self.player_grid

    def sample_action(self):
        return random.randint(0,self.xdim*self.ydim-1)
        #0-xdim*ydim





def handle_input(xdim,ydim):
    inputtext = input("next operation?")
    try:
        x = int(inputtext[0:2])
        y = int(inputtext[2:4])
        if (x>= xdim or y >= ydim):
            good_input=False
        else:
            #so y starts at the bottom
            y = (ydim-1)-y
            good_input = True
    except:
        x=0
        y=0
        good_input = False
    return x,y,good_input




if __name__ == '__main__':
    #main
    xdim = 9
    ydim = 9
    total_mines = 10

    safe_spots = xdim*ydim -total_mines
    good_input = False
    minecount= total_mines
    game = MinesweeperEnv(xdim,ydim,total_mines)
    game.reset()
    endstate = False
    game.print_player_grid()
    while not endstate:
        x,y,good_input = handle_input(xdim,ydim)
        if not good_input:
            continue
        action = x*ydim+y
        print("action",action)
        #reward,endstate,_ = game.open(x,y)
        _,reward,endstate,won = game.step(action)
        print(reward)
        game.print_player_grid()


    #endscreen
    if game.safe_spots != 0:
        print("BETTER LUCK NEXT TIME!")
    else:
        print("YOU WON!")
    time.sleep(3)



