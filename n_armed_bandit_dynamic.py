# This function simulates the performance of 4 approaches at solving the n armed bandit problem
# Problem specs are Sutton Book, p.42


# import libraries
import numpy as np
import operator
import matplotlib.pyplot as plt
from random_walk import *

# Set simulation parameters
n_bins  =   1000
n_iter  =   1000
n_arms  =   10
avg_R1  =   np.zeros([n_iter,n_bins])
avg_R2  =   np.zeros([n_iter,n_bins])
avg_R3  =   np.zeros([n_iter,n_bins])
avg_R4  =   np.zeros([n_iter,n_bins])
avg_R5  =   np.zeros([n_iter,n_bins])
avg_R6  =   np.zeros([n_iter,n_bins])

rW = random_walk(n_arms, n_bins, range(0, 1000, 500), 2)

# Loop on all iterations
for hh in range(0, n_iter):

    # Create bandit
    q_star  =   np.random.normal(0, 1, [1, n_arms])
    q_starD =   np.reshape(q_star, [n_arms, 1]) + rW

    # Rescale new q values
    q_starD =   np.divide(q_starD - np.min(np.min(q_starD)), np.max(np.max(q_starD)) - np.min(np.min(q_starD)) )
    q_starD =   q_starD * (np.max(q_star)-np.min(q_star)) + np.min(q_star)

    # Stationary methods
    q_estim1 =  np.random.normal(0, .0001, [1, n_arms])
    n_visit1 =  np.zeros([1, n_arms])
    q_estim2 =  np.random.normal(0, .0001, [1, n_arms])
    n_visit2 =  np.zeros([1, n_arms])
    q_estim3 =  np.random.normal(0, .0001, [1, n_arms])
    n_visit3 =  np.zeros([1, n_arms])

    # Non-stationary methods
    alpha    = 0.8;
    q_estim4 = np.random.normal(0, .0001, [1, n_arms])
    q_estim5 = np.random.normal(0, .0001, [1, n_arms])
    q_estim6 = np.random.normal(0, .0001, [1, n_arms])

    # Loop on time samples
    for ii in range(0, n_bins):
        # Select an action
        # --- method1 : greedy
        id_sel =   np.argmax(q_estim1) # select arm with highest q
        reward =   q_starD[id_sel,ii] + np.random.normal(0, 1, 1) # draw reward q(a) + random number
        n_visit1[0,id_sel] += 1
        q_estim1[0,id_sel] = q_estim1[0,id_sel] + (reward - q_estim1[0,id_sel])/n_visit1[0,id_sel]
        avg_R1[hh,ii] = reward

        # --- method2 : e-greedy 0.01
        if np.random.uniform(0,1)>=0.99:
            id_sel  =   np.random.randint(0,n_arms)
        else:
            id_sel  =   np.argmax(q_estim2)  # select arm with highest q
        reward = q_starD[id_sel,ii] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        n_visit2[0, id_sel] += 1
        q_estim2[0,id_sel] = q_estim2[0,id_sel] + (reward - q_estim2[0,id_sel])/n_visit2[0,id_sel]
        avg_R2[hh, ii] = reward

        # --- method3 : e-greedy 0.1
        if np.random.uniform(0, 1) >= 0.9:
            id_sel = np.random.randint(0, n_arms)
        else:
            id_sel = np.argmax(q_estim3)  # select arm with highest q
        reward = q_starD[id_sel,ii] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        n_visit3[0, id_sel] += 1
        q_estim3[0, id_sel] = q_estim3[0,id_sel] + (reward - q_estim3[0,id_sel])/n_visit3[0,id_sel]
        avg_R3[hh, ii] = reward

        # --- method4 : greedy - non-stationary
        id_sel = np.argmax(q_estim4)  # select arm with highest q
        reward = q_starD[id_sel, ii] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        q_estim4[0, id_sel] = q_estim4[0, id_sel] + (reward - q_estim4[0, id_sel]) * alpha
        avg_R4[hh, ii] = reward

        # --- method5 : e-greedy 0.01 - non-stationary
        if np.random.uniform(0, 1) >= 0.99:
            id_sel = np.random.randint(0, n_arms)
        else:
            id_sel = np.argmax(q_estim5)  # select arm with highest q
        reward = q_starD[id_sel, ii] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        q_estim5[0, id_sel] = q_estim5[0, id_sel] + (reward - q_estim5[0, id_sel]) * alpha
        avg_R5[hh, ii] = reward

        # --- method6 : e-greedy 0.1 - non-stationary
        if np.random.uniform(0, 1) >= 0.9:
            id_sel = np.random.randint(0, n_arms)
        else:
            id_sel = np.argmax(q_estim6)  # select arm with highest q
        reward = q_starD[id_sel, ii] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        q_estim6[0, id_sel] = q_estim6[0, id_sel] + (reward - q_estim6[0, id_sel]) / * alpha
        avg_R6[hh, ii] = reward




# display
plt.plot( [500, 500], [0, 1.5], 'r--')
plt.plot( np.mean(avg_R1,0), 'g', label='e=0 - greedy')
plt.plot( np.mean(avg_R2,0), 'r', label='e=0.01 - +expl')
plt.plot( np.mean(avg_R3,0) ,'k', label='e=0.1 - ++expl')
plt.plot( np.mean(avg_R4,0), 'g--', label='e=0 - greedyNS')
plt.plot( np.mean(avg_R5,0), 'r--', label='e=0 - +explNS')
plt.plot( np.mean(avg_R6,0), 'k--', label='e=0 - ++explNS')
plt.ylabel('Average reward')
plt.legend(loc='upper left')
plt.xlabel('Nb of plays')
axes = plt.gca()
axes.annotate('random walk', xy=(500, 1), xytext=(650, 1.3), arrowprops=dict(facecolor='black', shrink=0.05),)
axes.set_ylim([0, 1.5])
plt.show()