""" This script implements a comparison scheme between different types of learning
    Each learning algorithm is ran several times """

import pickle
import numpy as np
from ReinforcementLearning.blackJack.blackjack_policy_solver_light import *


# Set variables
learningTypes   =   ['MC_egreedy0', 'MC_egreedy0.05', 'MC_egreedy0.1', 'TD_egreedy0', 'TD_egreedy0.05', 'TD_egreedy0.1']
iterVec         =   range(0, 500000, 10000)
iterSteps       =   [0]+[2000]*50
nIterations     =   50
svFile          =   'BlackJack_comparison_TDgreedy_100Kiter'

# Load ground truth
groundT         =   pickle.load( open('ReinforcementLearning/blackJack/blackjack_solution.p', 'rb') )['policy']

# Set empty containers
MSE     =   np.zeros([nIterations, len(iterSteps), len(learningTypes)])

print('Launching learning evaluation script for 3 type of learning:')
# Loop on number of iterations
for nIt in range(nIterations):

    print('\t***Iteration: ', str(nIt))
    # Initialize the solvers - MC
    solver_eg0      =   blackjack_policy_solver_light(initMode='random', eGreedy=0.0)
    solver_eg005    =   blackjack_policy_solver_light(initMode='random', eGreedy=0.05)
    solver_eg01     =   blackjack_policy_solver_light(initMode='random', eGreedy=0.1)
    # Initialize the solvers - TD
    solver_td0_eg0  =   blackjack_policy_solver_light(initMode='random', eGreedy=0, learnRate=0.1)
    solver_td0_eg005=   blackjack_policy_solver_light(initMode='random', eGreedy=0.05, learnRate=0.1)
    solver_td0_eg01 =   blackjack_policy_solver_light(initMode='random', eGreedy=0.1, learnRate=0.1)

    for nSt in range(len(iterSteps)):

        print('\t\tSolution step: ', str(nSt))
        # Solve for policy - MC
        #solver_eg0.monte_carlo_ES(iterSteps[nSt])
        #solver_eg005.monte_carlo_ES(iterSteps[nSt])
        #solver_eg01.monte_carlo_ES(iterSteps[nSt])
        # Solve for policy - TD
        solver_td0_eg0.TD0(iterSteps[nSt])
        solver_td0_eg005.TD0(iterSteps[nSt])
        solver_td0_eg01.TD0(iterSteps[nSt])

        # Compute MSE - MC
        #MSE[nIt, nSt, 0]    =   np.sqrt(np.sum((solver_eg0.policy_agent - groundT)**2))
        #MSE[nIt, nSt, 1]    =   np.sqrt(np.sum((solver_eg005.policy_agent - groundT) ** 2))
        #MSE[nIt, nSt, 2]    =   np.sqrt(np.sum((solver_eg01.policy_agent - groundT) ** 2))
        MSE[nIt, nSt, 3]    =   np.sqrt(np.sum((solver_td0_eg0.policy_agent - groundT) ** 2))
        MSE[nIt, nSt, 4]    =   np.sqrt(np.sum((solver_td0_eg005.policy_agent - groundT) ** 2))
        MSE[nIt, nSt, 5]    =   np.sqrt(np.sum((solver_td0_eg01.policy_agent - groundT) ** 2))


# Save to file
pack    =   {'lerningTypes':learningTypes, 'iterSteps':iterSteps, 'groundT':groundT, 'MSE':MSE}
pickle.dump( pack, open('ReinforcementLearning/blackJack/'+svFile+'.p', 'wb') )


# Print results
avgMSE  =   np.mean(MSE, axis=0)
plt.figure()
plt.plot(avgMSE[:,0], 'g', label='Exploratory starts')
plt.plot(avgMSE[:,1], 'r', label='e-greedy: 0.05')
plt.plot(avgMSE[:,2], 'k', label='e-greedy: 0.1')
plt.plot(avgMSE[:,3], 'g', label='Exploratory starts')
plt.plot(avgMSE[:,4], 'r', label='e-greedy: 0.05')
plt.plot(avgMSE[:,5], 'k', label='e-greedy: 0.1')
ax  =   plt.gca()
ax.set_xlim([0,50])
plt.legend(loc='upper right')
ax.set_xticklabels( [int(x) for x in ax.get_xticks()*2000] )
ax.set_xlabel('Number of games')
ax.set_ylabel('Distance to optimal policy (MSE)')