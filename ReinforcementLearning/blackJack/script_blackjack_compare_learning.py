""" This script implements a comparison scheme between different types of learning
    Each learning algorithm is ran several times """

import pickle
import numpy as np
from ReinforcementLearning.blackJack.blackjack_policy_solver_light import *


# Set variables
learningTypes   =   ['egreedy0', 'egreedy0.05', 'egreedy0.1']
iterVec         =   range(0, 500000, 10000)
iterSteps       =   [0]+[2000]*50
nIterations     =   50
svFile          =   'results_compare_learning_egreedy_500K'

# Load ground truth
groundT         =   pickle.load( open('ReinforcementLearning/blackJack/blackjack_solution.p', 'rb') )['policy']

# Set empty containers
MSE     =   np.zeros([nIterations, len(iterSteps), len(learningTypes)])

print('Launching learning evaluation script for 3 type of learning:')
# Loop on number of iterations
for nIt in range(nIterations):

    print('\t***Iteration: ', str(nIt))
    # Initialize the solvers
    solver_eg0      =   blackjack_policy_solver_light(initMode='random', eGreedy=0.0)
    solver_eg005    =   blackjack_policy_solver_light(initMode='random', eGreedy=0.05)
    solver_eg01     =   blackjack_policy_solver_light(initMode='random', eGreedy=0.1)

    for nSt in range(len(iterSteps)):

        print('\t\tSolution step: ', str(nSt))
        # Solve for policy
        solver_eg0.monte_carlo_ES(iterSteps[nSt])
        solver_eg005.monte_carlo_ES(iterSteps[nSt])
        solver_eg01.monte_carlo_ES(iterSteps[nSt])

        # Compute MSE
        MSE[nIt, nSt, 0]    =   np.sqrt(np.sum((solver_eg0.policy_agent - groundT)**2))
        MSE[nIt, nSt, 1]    =   np.sqrt(np.sum((solver_eg005.policy_agent - groundT) ** 2))
        MSE[nIt, nSt, 2]    =   np.sqrt(np.sum((solver_eg01.policy_agent - groundT) ** 2))


# Save to file
pack    =   {'lerningTypes':learningTypes, 'iterSteps':iterSteps, 'groundT':groundT, 'MSE':MSE}
pickle.dump( pack, open('ReinforcementLearning/blackJack/'+svFile+'.p', 'wb') )


# Print results
avgMSE  =   np.mean(MSE, axis=0)
plt.figure()
plt.plot(avgMSE[:,0], 'g', label='Exploratory starts')
plt.plot(avgMSE[:,1], 'r', label='e-greedy: 0.05')
plt.plot(avgMSE[:,2], 'k', label='e-greedy: 0.1')
ax  =   plt.gca()
ax.set_xlim([0,50])
plt.legend(loc='upper right')
ax.set_xticklabels( [int(x) for x in ax.get_xticks()*2000] )
ax.set_xlabel('Number of games')
ax.set_ylabel('Distance to optimal policy (MSE)')