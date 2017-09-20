# This function simulates the performance of 4 approaches at solving the n armed bandit problem
# Problem specs are Sutton Book, p.42


# import libraries
import numpy as np
import operator
import matplotlib.pyplot as plt

# Set simulation parameters
n_bins  =   1000
n_iter  =   2000
n_arms  =   10
q_table =   np.zeros([n_arms,n_bins])
avg_R1  =   np.zeros([n_iter,n_bins])
avg_R2  =   np.zeros([n_iter,n_bins])
avg_R3  =   np.zeros([n_iter,n_bins])


# Loop on all iterations
for hh in range(0, n_iter):

    # Create bandit
    q_star  =   np.random.normal(0, 1, [1, n_arms])

    # Loop on time samples
    for ii in range(0, n_bins):
        # Select an action
        # --- method1 : greedy
        sumQ    =   np.sum(q_table[:,0:ii],1)
        id_sel, value  =   max( enumerate(sumQ), key=operator.itemgetter(1) ) # select arm with highest q
        val_sel =   q_star[0,id_sel] + np.random.normal(0, 1, 1) # draw reward q(a) + random number
        q_table[id_sel,ii] = val_sel
        avg_R1[hh, ii] = sumQ[id_sel]/ii

        # --- method2 : e-greedy 0.01
        sumQ = np.sum(q_table[:, 0:ii], 1)
        if np.random.uniform(0,1)>=0.99:
            id_sel  =   np.random.randint(0,n_arms)
        else:
            id_sel, value = max(enumerate(sumQ), key=operator.itemgetter(1))  # select arm with highest q
        val_sel = q_star[0, id_sel] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        q_table[id_sel, ii] = val_sel
        avg_R2[hh, ii] = sumQ[id_sel] / ii

        # --- method3 : e-greedy 0.1
        sumQ = np.sum(q_table[:, 0:ii], 1)
        if np.random.uniform(0, 1) >= 0.9:
            id_sel = np.random.randint(0, n_arms)
        else:
            id_sel, value = max(enumerate(sumQ), key=operator.itemgetter(1))  # select arm with highest q
        val_sel = q_star[0, id_sel] + np.random.normal(0, 1, 1)  # draw reward q(a) + random number
        q_table[id_sel, ii] = val_sel
        avg_R3[hh, ii] = sumQ[id_sel] / ii

# display
plt.plot( np.mean(avg_R1,0), 'g')
plt.plot( np.mean(avg_R2,0), 'r')
plt.plot( np.mean(avg_R3,0) ,'k' )
plt.ylabel('Average reward')
plt.show()