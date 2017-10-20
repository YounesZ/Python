""" This script implements a blackjack game between a player (human user) and an agent implementing different forms of
    Reinforcement Learning to compare how fast these forms learn optimal policy"""

import numpy as np
from ReinforcementLearning.blackJack.blackjack_policy_solver_light import *
from copy import deepcopy


# Game history
nGames          =   1000
GameHistory     =   np.zeros([1, nGames])

# Loop and play
for ipl in range(nGames):

    # Player's turn, doesn't really matter who goes first


    # Agent's turn
    bjk_player  =   blackjack_policy_solver_light()
    bjk_player.init_episode()
    dealer      =   bjk_player.dealer
    bjk_player.run_episode()




