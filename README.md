Reinforcement Learning Minesweeper Solver


My project idea is to program a reinforcement learning system that plays the game Minesweeper. For this I intend to implement a simple Q-learning system like in the stanford paper linked below, and try to improve its results (bring your own method project type).
To gather data I will program a simple version of the game. 

So the first step of the project will be to program a minimal version of minesweeper, which should not take more than ~2 days.
The next step will be implementing the Q-learning system, which I expect to be the longest part of the project.
After that i will try to find ways to improve the implementation. The workload of this will be quite dependent on the results.
The final step would be to program a more user friendly version of the game to better show the results, the required time for this is heavily dependent on how long the previous steps take.



relevant papers:
https://cs229.stanford.edu/proj2015/372_report.pdf
https://pure.tue.nl/ws/portalfiles/portal/307404348/Thesis_BDS_Juli_G._Smulders.pdf


########################################################
ASSIGNMENT 2:

Implementation:
requirements.txt:
you need pytorch and numpy.
i just wrote which versions i have installed, i don't know which other versions work.

Gym:
for easier modifications, and because i was missing some features in existing implementation, i wrote a gym myself.

Networks:
currently there are two different implementations, where the only difference is the used Neural Network. One of them uses 3 fully connected layers and the other one uses two Convolutional layers and then two fully connected layers.
other than that both implementations are equal.
They use a replay memory with random sampling, use the AdamW optimizer, SoothL1Loss and clipping of gradients, and a policy_net and target_net, for more stable training.

Implementation Notes:
reloading checkpoints does not safe replay memory.
probably doesn't matter too much.


How To Use:
files:
the files were split up to allow training different models at the same time more easily.

minesweeper_AI_InfiniteTrainingCNN.py
A python script which trains a Convolutional Neural Net as long as it is running. The weights and other parameters are saved every 1000 episodes (=played game boards)

minesweeper_AI_InfiniteTraining.py
A python script which trains a non-convolutional Neural Net as long as it is running. The weights and other parameters are saved every 1000 episodes (=played game boards)

minesweeper_AI.py
(probably out of date, use one of the above)
A python script which trains either a Convolutional or non-convolutional Neural Net (depending on a variable)

test.py
a test script, which loads the saved states and then shows some training data (like average reward of every 100 episodes).
Then it plays a certain amount of games slowly and with UI so a User can see what it is doing.
Then it plays 10000 games in the background and just prints the win rate when done.

minesweeper.py
the gym. rewards can be changed in init of the class.



Training instructions:
	Start from scratch:
	at the top of the script select which grid size you want to train on.
	then run the script and enter 'n' when prompted if a last state should be loaded
	
	Continue Training:
	the savefiles are located in the savefiles or savefilesCNN folder. if you want to load a different savefile, you can copy and paste all corresponding files in there (for older safefiles which might miss some safed parameters, creating an empty file should work to get it running anyways)
	at the top of the script select which grid size you want to train on (needs to match the size the training started on)
	set i_episode to the episode you want to continue training with (warning, all savefiles after this will be overwritten as training progresses)
	run the script and enter 'y' when prompted if a last state should be loaded
	
Testing instructions:
	set small_field and i_episode like when continuing training.
	set conv_or_not to choose whether you want the CNN-> True or the NN->False. 
	set not_random to True if you want to test a model, to False if you want to create a baseline with random moves
	set visual_game_amount and move_delay to allow you to see how the model plays games
	run the script

for your convenience some pre-trained models from the last run (using parameters as they are now) are provided in savefiles and savefilesCNN (also in other_runs, but those might have incomplete data).






Goals:
The primary metric is the win rate.
To get an estimate of the win rate we let the model solve 10000 boards.

There are 3 Milestones regarding the win rate:
1) noticeably beat random guessing
2) beat other Deep Q Learning approaches (from relevant papers)
3) equal statistical solvers (so ~90% for the big board)
4) never hit a mine (not possible if every game has to be played to the end)

The reasonable target would be #2, to beat the other Q Learning models from the mentioned papers.




Current Standing:
For now comparison to the paper isn't easily done, as i have mainly used other board sizes and metrics, in an attempt to see progress in any way.
What is safe to say is that both the current models do not significantly beat random guessing, and do not noticeably improve over time, as can be seen in the result of some test runs in comparison.txt.



Time Breakdown:
(rough estimates, a lot of work has also been done while focused on other things(e.g. training, testing and hyperparameter tuning, every now and then). this time is not included here)
Program Gym: ~4h
Learn Pytorch: 5h
Write initial Deep Q Learning Network: 4h
Continuous improvements to the networks and especially usability + hyperparameter tuning: 12h



Considerations to continue:
merge all the training and test files for better user experience.
maybe scale down the field to 4x4 with 3 mines, to match the Stanford paper.
changing architecture of the networks is also a serious consideration.











