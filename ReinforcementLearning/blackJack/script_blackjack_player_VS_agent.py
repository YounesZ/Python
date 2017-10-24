""" This script implements a blackjack game between a player (human user) and an agent implementing different forms of
    Reinforcement Learning to compare how fast these forms learn optimal policy

    TD DO:
        - implement finite card deck: need to synchronize card probabilities among the 2 classes

"""

import numpy as np
import matplotlib.pyplot as plt
from ReinforcementLearning.blackJack.blackjack_policy_solver_light import *
from Utils.games.blackjack import *
from copy import deepcopy


# Game history
nGames          =   1000
GameHistory     =   np.zeros([1, nGames])
cardConv        =   { str(_key) : _key for _key in range(2,11)}
cardConv.update({'1':11, 'jack':10, 'queen':10, 'king':10})

# Init the agents
bjk_agent       =   blackjack_policy_solver_light()

# Init the display
# --- Initial line
minY    =   -10
maxY    =   10

plt.ion()
fig     =   plt.figure()
ax1     =   fig.add_subplot(111)
ax1.set_xlim([0, nGames])
ax1.set_ylim([minY, maxY])
line1,  =   ax1.plot([], [], 'b-')

#plt.plot([0, nGames], [0, 0], '--r')


# Loop and play
ipl     =   0
while ipl<nGames:

    # ========
    # Player's turn, doesn't really matter who goes first
    ipl         +=  1
    bjk_player  =   blackJack()
    status      =   2
    while status>1 and bjk_player.turn=='agent':
        # Action?
        action  =   input('Player action: (h)it, (s)tick or e(x)it?')
        if action   ==  'h':
            bjk_player.hand_do('hit', statUpd=False)
        elif action ==  's':
            bjk_player.hand_do('stick', statUpd=False)
        elif action ==  'x':
            ipl =   nGames
            break
        status  =   bjk_player.game_status(statusOnly=True, printStatus=True)

    # ========
    # Agent1's turn
    agC         =   cardConv[bjk_player.dealer['hand'][0].split('_')[0]]
    dlC         =   cardConv[bjk_player.agent['hand'][0].split('_')[0]]
    bjk_agent.init_episode2(A=[agC], D=[dlC])
    bjk_agent.monte_carlo(nIterations=1, dealerFinalScore=bjk_player.agent['value'])

    # ========
    # Update plot
    minY    =   min(minY, sum(bjk_agent.history))
    maxY    =   max(maxY, sum(bjk_agent.history))
    line1.set_xdata(np.append(line1.get_xdata(), ipl))
    line1.set_ydata(np.append(line1.get_ydata(), sum(bjk_agent.history)))
    ax1.set_ylim([minY, maxY])
    fig.canvas.draw()

