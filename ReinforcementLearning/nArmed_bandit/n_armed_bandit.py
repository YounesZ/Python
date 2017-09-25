# This function simulates the performance of 4 approaches at solving the n armed bandit problem
# Problem specs are Sutton Book, p.42


# import libraries
import numpy as np
import operator
import matplotlib.pyplot as plt

# Set simulation parameters
n_bins  =   1000
n_iter  =   1000
n_arms  =   10
avg_R1  =   np.zeros([n_iter,n_bins])
avg_R2  =   np.zeros([n_iter,n_bins])
avg_R3  =   np.zeros([n_iter,n_bins])


# Loop on all iterations
for hh in range(0, n_iter):

    # Create bandit
    q_star   =  np.random.normal(0, 1, [1, n_arms])
    q_estim1 =  np.random.normal(0, .0001, [1, n_arms])
    n_visit1 =  np.zeros([1, n_arms])
    q_estim2 =  np.random.normal(0, .0001, [1, n_arms])
    n_visit2 =  np.zeros([1, n_arms])
    q_estim3 =  np.random.normal(0, .0001, [1, n_arms])
    n_visit3 =  np.zeros([1, n_arms])


    # Loop on time samples
    for ii in range(0, n_bins):
        # Select an action
        # --- method1 : greedy
        id_sel =   np.argmax(q_estim1) # select arm with highest q
        reward =   q_star[0,id_sel] + np.random.normal(0, 1, 1) # draw reward q(a) + random number
        n_visit1[0,id_sel] += 1
        q_estim1[0,id_sel] = q_estim1[0,id_sel] + (reward - q_estim1[0,id_sel])/n_visit1[0,id_sel]
        avg_R1[hh,ii] = reward

        # --- method2 : e-greedy 0.01
        if np.random.uniform(0,1)>=0.99:
            id_sel  =   np.random.randint(0,n_arms)
        else:
            id_sel  =   np.argmax(q_estim2)  # select arm with highest q
        reward = q_star[0, id_sel] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        n_visit2[0, id_sel] += 1
        q_estim2[0,id_sel] = q_estim2[0,id_sel] + (reward - q_estim2[0,id_sel])/n_visit2[0,id_sel]
        avg_R2[hh, ii] = reward

        # --- method3 : e-greedy 0.1
        if np.random.uniform(0, 1) >= 0.9:
            id_sel = np.random.randint(0, n_arms)
        else:
            id_sel = np.argmax(q_estim3)  # select arm with highest q
        reward = q_star[0, id_sel] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        n_visit3[0, id_sel] += 1
        q_estim3[0, id_sel] = q_estim3[0,id_sel] + (reward - q_estim3[0,id_sel])/n_visit3[0,id_sel]
        avg_R3[hh, ii] = reward

# display
plt.plot( np.mean(avg_R1,0), 'g')
plt.plot( np.mean(avg_R2,0), 'r')
plt.plot( np.mean(avg_R3,0) ,'k' )
plt.ylabel('Average reward')
plt.show()