# This function simulates the performance of 4 approaches at solving the n armed bandit problem
# Problem specs are Sutton Book, p.42


# import libraries
import numpy as np
import operator
import matplotlib.pyplot as plt
from Utils.programming import ut_random_walk
from itertools import combinations


print('\n\n\t***\tSTARTED simulating greedy offensive line selection:')

# Set simulation parameters
n_bins  =   10000
n_iter  =   1000
n_arms  =   12
n_pulls =   3
alpha   =   0.8
avg_R1  =   np.zeros([n_iter,n_bins])
avg_R2  =   np.zeros([n_iter,n_bins])
avg_R3  =   np.zeros([n_iter,n_bins])

#rW = ut_random_walk.random_walk(n_arms, n_bins, range(0, 1000, 500), 2)

# Make random combinations
lsLines =   list( combinations( range(n_arms), n_pulls ) )
priorL  =   [(0,1,2), (3,4,5), (6,7,8), (9,10,11)]
priorIx =   [0] * len(priorL)               # holder for the indices of the prior lines
newlIx  =   list( range(len(lsLines)) )     # holder for the indices of the new lines
lineV   =   [0] * len(lsLines)              # true value of each offensive line (i.e. combination of players)
for ii in range(len(priorL)):
    compP   =   [x==priorL[ii] for x in lsLines]
    idx     =   np.where(compP)[0][0]
    priorIx[ii] =   idx
    newlIx.pop( np.where(newlIx==idx)[0][0] )
    lineV[idx]  =   1 - 0.33*ii

# Loop on all iterations : each iteration is a new bandits configuration
for hh in range(0, n_iter):

    print('\tIteration number: ', str(hh), '/', str(n_iter), ': ', end="")

    # Create bandit
    q_star  =   lineV
    for jj  in newlIx:
        q_star[jj]  =   np.random.normal(0,1,1)[0]

    # Static bandits
    q_estim1 =  np.random.normal(0, .0001, [1, len(lsLines)])
    q_estim1[0, priorIx]    =   list( 1 - np.multiply( 0.33, range(4)) )
    n_visit1 =  np.zeros([1, len(lsLines)])
    q_estim2 =  np.random.normal(0, .0001, [1, len(lsLines)])
    q_estim1[0, priorIx] = list(1 - np.multiply(0.33, range(4)))
    n_visit2 =  np.zeros([1, len(lsLines)])
    q_estim3 =  np.random.normal(0, .0001, [1, len(lsLines)])
    q_estim1[0, priorIx] = list(1 - np.multiply(0.33, range(4)))
    n_visit3 =  np.zeros([1, len(lsLines)])

    # Loop on time samples
    for ii in range(0, n_bins):
        # Select an action
        # --- method1 : greedy
        id_sel =   np.argmax(q_estim1) # select arm with highest q
        reward =   q_star[id_sel] + np.random.normal(0, 1, 1) # draw reward q(a) + random number
        n_visit1[0,id_sel] += 1
        q_estim1[0,id_sel] = q_estim1[0,id_sel] + (reward - q_estim1[0,id_sel])/n_visit1[0,id_sel]
        avg_R1[hh,ii] = reward

        # --- method2 : e-greedy 0.01
        if np.random.uniform(0,1)>=0.99:
            id_sel  =   np.random.randint(0,n_arms)
        else:
            id_sel  =   np.argmax(q_estim2)  # select arm with highest q
        reward = q_star[id_sel] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        n_visit2[0, id_sel] += 1
        q_estim2[0,id_sel] = q_estim2[0,id_sel] + (reward - q_estim2[0,id_sel])/n_visit2[0,id_sel]
        avg_R2[hh, ii] = reward

        # --- method3 : e-greedy 0.1
        if np.random.uniform(0, 1) >= 0.9:
            id_sel = np.random.randint(0, n_arms)
        else:
            id_sel = np.argmax(q_estim3)  # select arm with highest q
        reward = q_star[id_sel] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        n_visit3[0, id_sel] += 1
        q_estim3[0, id_sel] = q_estim3[0,id_sel] + (reward - q_estim3[0,id_sel])/n_visit3[0,id_sel]
        avg_R3[hh, ii] = reward

        if (ii+1)/n_bins*100 == int((ii+1)/n_bins*100):
            print(str((ii+1)/n_bins*100), '%, ', end="")
    print('done.')



# display
plt.plot( [500, 500], [0, 1.5], 'r--')
plt.plot( np.mean(avg_R1,0), 'g', label='e=0 - greedy')
plt.plot( np.mean(avg_R2,0), 'r', label='e=0.01 - +expl')
plt.plot( np.mean(avg_R3,0) ,'k', label='e=0.1 - ++expl')
plt.ylabel('Average reward')
plt.legend(loc='lower right')
plt.xlabel('Nb of plays')
axes = plt.gca()
#axes.annotate('random walk', xy=(500, 0.2), xytext=(100, 0.2), arrowprops=dict(facecolor='black', shrink=0.05),)
axes.set_ylim([0.5, 1.8])
plt.show()