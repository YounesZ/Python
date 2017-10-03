# This function simulates the performance of 4 approaches at solving the n armed bandit problem
# Problem specs are Sutton Book, p.42


# import libraries
import numpy as np
import operator
import matplotlib.pyplot as plt
from Utils.programming import ut_random_walk
from Utils.maths import ut_poisson_proba
from itertools import combinations


print('\n\n\t***\tSTARTED simulating greedy offensive line selection:')

# Set simulation parameters
n_iter  =   1000
n_arms  =   12
n_pulls =   3
alpha   =   0.8

# Set decay and recovery functions
decay   =   lambda ltd: np.exp(-ltd/186)
recovery=   lambda ltr: np.log(ltr)*0.7/np.log(300)

# Sample line changes: hard-coded median line shift = 45s
# https://www.sportingcharts.com/nhl/stats/average-ice-time-per-shift/2016/
changeT =   []
lambdaLS=   45      # expected line shifts after 45s
shiftL  =   np.array(range(15,121,15))/60*4
probaSL =   ut_poisson_proba.main( int(lambdaLS/60*4)+1, shiftL )[0]
while np.sum(changeT)<3600/15:
    rnd     =   np.random.normal(0,0.25,[1, len(probaSL)])
    prbrnd  =   np.argmax( probaSL + rnd )
    changeT.append(shiftL[prbrnd])

# Initialize containers
avg_R1  =   np.zeros([n_iter,len(changeT)])
avg_R2  =   np.zeros([n_iter,len(changeT)])
avg_R3  =   np.zeros([n_iter,len(changeT)])

for hh in range(n_iter):

    print('\tIteration ', str(hh), '/', str(n_iter), ': ', end="")

    # Make random combinations: lineups
    lsLines =   list( combinations( range(n_arms), n_pulls ) )
    priorL  =   [(0,1,2), (3,4,5), (6,7,8), (9,10,11)]
    priorIx =   [0] * len(priorL)               # holder for the indices of the prior lines
    lineV   =   np.random.normal(0.05, .25, [1, len(lsLines)])              # true value of each offensive line (i.e. combination of players)
    for ii in range(len(priorL)):
        compP   =   [x==priorL[ii] for x in lsLines]
        idx     =   np.where(compP)[0][0]
        priorIx[ii] =   idx
        lineV[0,idx]=   1 - 0.1*ii

    # Static bandits
    # --- Bandit1
    q_estim1 = np.random.normal(0.05, .25, [1, len(lsLines)])
    q_estim1[0, priorIx] = list(1 - np.multiply(0.1, range(4)))
    n_visit1 = np.zeros([1, len(lsLines)])
    # --- Bandit2
    q_estim2 = np.random.normal(0.05, .25, [1, len(lsLines)])
    q_estim2[0, priorIx] = list(1 - np.multiply(0.1, range(4)))
    n_visit2 = np.zeros([1, len(lsLines)])
    # --- Bandit3
    q_estim3 = np.random.normal(0.05, .25, [1, len(lsLines)])
    q_estim3[0, priorIx] = list(1 - np.multiply(0.1, range(4)))
    n_visit3 = np.zeros([1, len(lsLines)])

    ### PASS1 : GREEDY METHOD
    # Simulate value changes: players and lines
    playerFitness   =   np.ones([2, n_arms+1])
    playerRested    =   np.zeros([1, n_arms+1])
    selectedLine    =   []
    for jj in range(len(changeT)):
        # Heuristic on line value
        pF          =   playerFitness + (playerFitness[0, :] - playerFitness[1, :]) * recovery(np.minimum(300,np.maximum(1, playerRested[0, :])))
        lF          =   [np.prod(pF[0,x]) for x in lsLines]
        # Select line --- method1 : greedy
        id_sel      =   np.argmax( np.multiply(q_estim1, np.reshape(lF, [1, len(lsLines)])) )  # select arm with highest q
        selectedLine.append(id_sel)
        # Update players: fitness
        playersOn   =   lsLines[id_sel]
        for kk in playersOn:
            playerFitness[0,kk]     =   playerFitness[1,kk] + (playerFitness[0,kk]-playerFitness[1,kk])*recovery(max(1,playerRested[0,kk]))
        # Update line value
        lineValue   =   lineV[0,id_sel] * np.prod(playerFitness[0,playersOn])
        # Compute reward
        reward      =   lineValue + np.random.normal(0, .005, 1)  # draw reward q(a) + random number
        # Update line value estimate
        n_visit1[0, id_sel] += 1
        q_estim1[0, id_sel] = q_estim1[0, id_sel] + (reward - q_estim1[0, id_sel]) / n_visit1[0, id_sel]
        avg_R1[hh, jj] = reward
        # Update players: fatigue
        for kk in playersOn:
            playerFitness[1, kk] = playerFitness[0, kk] * decay(changeT[jj]*15)
        # Update players: rest
        playerRested[0, playersOn] = -changeT[jj] * 15
        playerRested    +=  changeT[jj]*15

    ### PASS2 : e-GREEDY METHOD : epsilon=0.01
    # Simulate value changes: players and lines
    playerFitness   =   np.ones([2, n_arms+1])
    playerRested    =   np.zeros([1, n_arms+1])
    selectedLine    =   []
    for jj in range(len(changeT)):
        # Heuristic on line value
        pF          =   playerFitness + (playerFitness[0, :] - playerFitness[1, :]) * recovery(np.minimum(300,np.maximum(1, playerRested[0, :])))
        lF          =   [np.prod(pF[0,x]) for x in lsLines]
        # Select line --- method2 : e-greedy 0.01
        if np.random.uniform(0, 1) >= 0.99:
            id_sel  =   np.random.randint(0, len(lsLines))
        else:
            id_sel  =   np.argmax( np.multiply(q_estim1, np.reshape(lF, [1, len(lsLines)])) )  # select arm with highest q
        selectedLine.append(id_sel)
        # Update players: fitness
        playersOn   =   lsLines[id_sel]
        for kk in playersOn:
            playerFitness[0,kk]     =   playerFitness[1,kk] + (playerFitness[0,kk]-playerFitness[1,kk])*recovery(max(1,playerRested[0,kk]))
        # Update line value
        lineValue   =   lineV[0,id_sel] * np.prod(playerFitness[0,playersOn])
        # Compute reward
        reward      =   lineValue + np.random.normal(0, .005, 1)  # draw reward q(a) + random number
        # Update line value estimate
        n_visit1[0, id_sel] += 1
        q_estim1[0, id_sel] = q_estim1[0, id_sel] + (reward - q_estim1[0, id_sel]) / n_visit1[0, id_sel]
        avg_R2[hh, jj] = reward
        # Update players: fatigue
        for kk in playersOn:
            playerFitness[1, kk] = playerFitness[0, kk] * decay(changeT[jj]*15)
        # Update players: rest
        playerRested[0, playersOn] = -changeT[jj] * 15
        playerRested    +=  changeT[jj]*15

    ### PASS3 : e-GREEDY METHOD : epsilon=0.1
    # Simulate value changes: players and lines
    playerFitness   =   np.ones([2, n_arms+1])
    playerRested    =   np.zeros([1, n_arms+1])
    selectedLine    =   []
    for jj in range(len(changeT)):
        # Heuristic on line value
        pF          =   playerFitness + (playerFitness[0, :] - playerFitness[1, :]) * recovery(np.minimum(300,np.maximum(1, playerRested[0, :])))
        lF          =   [np.prod(pF[0,x]) for x in lsLines]
        # Select line --- method2 : e-greedy 0.01
        if np.random.uniform(0, 1) >= 0.9:
            id_sel  =   np.random.randint(0, len(lsLines))
        else:
            id_sel  =   np.argmax( np.multiply(q_estim1, np.reshape(lF, [1, len(lsLines)])) )  # select arm with highest q
        selectedLine.append(id_sel)
        # Update players: fitness
        playersOn   =   lsLines[id_sel]
        for kk in playersOn:
            playerFitness[0,kk]     =   playerFitness[1,kk] + (playerFitness[0,kk]-playerFitness[1,kk])*recovery(max(1,playerRested[0,kk]))
        # Update line value
        lineValue   =   lineV[0,id_sel] * np.prod(playerFitness[0,playersOn])
        # Compute reward
        reward      =   lineValue + np.random.normal(0, .005, 1)  # draw reward q(a) + random number
        # Update line value estimate
        n_visit1[0, id_sel] += 1
        q_estim1[0, id_sel] = q_estim1[0, id_sel] + (reward - q_estim1[0, id_sel]) / n_visit1[0, id_sel]
        avg_R3[hh, jj] = reward
        # Update players: fatigue
        for kk in playersOn:
            playerFitness[1, kk] = playerFitness[0, kk] * decay(changeT[jj]*15)
        # Update players: rest
        playerRested[0, playersOn] = -changeT[jj] * 15
        playerRested    +=  changeT[jj]*15

    print('done.')


# Plot the selected line along time
plt.figure();   plt.plot( [0, 60], [0, 0], 'r--', label='line1')
plt.plot( [0, 60], [136, 136], 'm--', label='line2')
plt.plot( [0, 60], [200, 200], 'y--', label='line3')
plt.plot( [0, 60], [219, 219], 'g--', label='line4')
plt.plot(np.cumsum(np.array(changeT)) * 15 / 60, selectedLine)
axes    =   plt.gca()
axes.set_xlim([0, 60])
plt.legend(loc='lower right')
plt.xlabel('Time on ice (m)')
plt.ylabel('Selected line (index)')
plt.figure(); plt.plot(avg_R1[0,:])


# Plot the average reward for each policy
plt.figure()
plt.plot( np.mean(avg_R1, axis=0), 'r', label='greedy-0' )
plt.plot( np.mean(avg_R2, axis=0), 'g', label='greedy-0.01' )
plt.plot( np.mean(avg_R3, axis=0), 'b', label='greedy-0.1' )
axes    =   plt.gca()
axes.set_xlim([0, 60])
axes.set_ylim([0, 1])
plt.title('Players recovery factor: ' + str(0.7))
plt.legend(loc='upper right')
plt.xlabel('Time on ice (m)')
plt.ylabel('Avg reward')
plt.plot([10, 10], [0, 1], 'k--')


