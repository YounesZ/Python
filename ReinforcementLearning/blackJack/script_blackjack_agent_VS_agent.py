""" This script implements a blackjack game between a player (human user) and an agent implementing different forms of
    Reinforcement Learning to compare how fast these forms learn optimal policy

    TD DO:
        - replace string card names by number 10 at line 39-40
        - implement finite card deck: need to synchronize card probabilities among the 2 classes

"""

import numpy as np
import matplotlib.pyplot as plt
from ReinforcementLearning.blackJack.blackjack_policy_solver_light import *
from Utils.games.blackjack import *
from copy import deepcopy


# Game history
filename        =   'blackjack_agentVSagent_log'
wrkRep          =   '/home/younesz/Documents/Simulations/blackjack/agentVSagent/ESvsES'
dataPack        =   {'Agent1_policy': {}, 'Agent2_policy': {}}
nGames          =   10000000
GameHistory     =   []
SimilarityUsable    =   []
SimilarityUnusable  =   []

# Init the agents
bjk_agent1      =   blackjack_policy_solver_light(eGreedy=0.05)
bjk_agent2      =   blackjack_policy_solver_light(eGreedy=0.1)

# Init the display
# --- Initial line
minY1, maxY1    =   (-10, 10)
minY2, maxY2    =   (0, 5)
# Figure1: cumulative summary
plt.ion()
fig1    =   plt.figure()
ax1     =   fig1.add_subplot(121)
ax1.set_xlim([0, nGames])
ax1.set_ylim([minY1, maxY1])
line1,  =   ax1.plot([], [], 'b')
ax1.set_xlabel('Number of games')
ax1.set_ylabel('Cumulated reward')
# Figure2: divergence
ax2     =   fig1.add_subplot(122)
ax2.set_xlim([0, nGames])
ax2.set_ylim([minY2, maxY2])
line2,  =   ax2.plot([], [], 'r', label='Usable ace')
line3,  =   ax2.plot([], [], 'm', label='Unusable ace')
ax2.set_xlabel('Number of games')
ax2.set_ylabel('Euclidian distance')
#plt.plot([0, nGames], [0, 0], '--r')


# Loop and play
ipl     =   0
while ipl<nGames:

    # ========
    # Agent1's turn
    ipl     +=  1
    bjk_agent1.init_episode2()
    _, state_chain1, action_chain1, usable_chain1   =   bjk_agent1.run_episode(dealerFinalScore=0)

    # ========
    # Agent2's turn
    agC = bjk_agent1.dealer['state']
    dlC = bjk_agent1.agent['hand'][0]
    bjk_agent2.init_episode2(A=[agC], D=[dlC])
    _, state_chain2, action_chain2, usable_chain2   =   bjk_agent2.run_episode(dealerFinalScore=0)

    # ========
    # Distribute reward
    reward  =   int(bjk_agent1.agent['state']>bjk_agent2.agent['state']) - int(bjk_agent1.agent['state']<bjk_agent2.agent['state'])
    # To agent1
    for st, ac, us in zip(state_chain1, action_chain1, usable_chain1):
        bjk_agent1.returns[st[0] - 12, st[1] - 2, us, ac]       +=  reward
        bjk_agent1.visits[st[0] - 12, st[1] - 2, us, ac]        +=  1
        # State value and policy
        bjk_agent1.action_value[st[0] - 12, st[1] - 2, us, ac]  =   bjk_agent1.returns[st[0] - 12, st[1] - 2, us, ac] / bjk_agent1.visits[st[0] - 12, st[1] - 2, us, ac]
        bjk_agent1.policy_agent[st[0] - 12, st[1] - 2, us]      =   bjk_agent1.greedy_choice(bjk_agent1.action_value[st[0] - 12, st[1] - 2, us, :])
    # To agent2
    for st, ac, us in zip(state_chain2, action_chain2, usable_chain2):
        bjk_agent2.returns[st[0] - 12, st[1] - 2, us, ac]       -=  reward
        bjk_agent2.visits[st[0] - 12, st[1] - 2, us, ac]        +=  1
        # State value and policy
        bjk_agent2.action_value[st[0] - 12, st[1] - 2, us, ac]  =   bjk_agent2.returns[st[0] - 12, st[1] - 2, us, ac] / bjk_agent2.visits[st[0] - 12, st[1] - 2, us, ac]
        bjk_agent2.policy_agent[st[0] - 12, st[1] - 2, us]      =   bjk_agent2.greedy_choice(bjk_agent2.action_value[st[0] - 12, st[1] - 2, us, :])

    # ========
    if not ipl%100 or ipl==1:
        # Update plot1
        GameHistory.append(reward)
        minY1   =   min(minY1, sum(GameHistory))
        maxY1   =   max(maxY1, sum(GameHistory))
        line1.set_xdata(np.append(line1.get_xdata(), ipl))
        line1.set_ydata(np.append(line1.get_ydata(), np.sum(GameHistory)))
        ax1.set_ylim([minY1, maxY1])
        ax1.set_xlim([0, min(ipl*2, nGames)])
        # Update plot2
        SimilarityUsable.append(np.sqrt(np.sum((bjk_agent1.policy_agent[:, :, 1] - bjk_agent2.policy_agent[:, :, 1]) ** 2)))
        SimilarityUnusable.append(np.sqrt(np.sum((bjk_agent1.policy_agent[:, :, 0] - bjk_agent2.policy_agent[:, :, 0]) ** 2)))
        maxY2 = max(maxY2, max(SimilarityUsable), max(SimilarityUnusable))
        line2.set_xdata(line1.get_xdata())
        line2.set_ydata(np.append(line2.get_ydata(), SimilarityUsable[-1]))
        line3.set_xdata(line1.get_xdata())
        line3.set_ydata(np.append(line3.get_ydata(), SimilarityUnusable[-1]))
        ax2.set_xlim([0, min(ipl * 2, nGames)])
        ax2.set_ylim([minY2, maxY2])
        ax2.legend(loc='lower right')
        fig1.canvas.draw()


    if not ipl % 10000 or ipl == 1:
        # Save to file
        dataPack['Agent1_policy']  =   bjk_agent1.policy_agent
        dataPack['Agent2_policy']  =   bjk_agent2.policy_agent
        with open(wrkRep+filename+'_iter'+str(ipl)+'.p', "wb") as f:
            pickle.dump(dataPack, f, pickle.HIGHEST_PROTOCOL)
    # Save history
    dataPack    =   {'History':GameHistory, 'Figure':fig1}
    with open(wrkRep + 'game_history.p', "wb") as f:
        pickle.dump(dataPack, f, pickle.HIGHEST_PROTOCOL)