""" This function generates a random walk for n processes """

import numpy as np

def random_walk(n_processes, duration, transitions, factor=1):

    rW  =   np.zeros([n_processes, duration])

    # Start walking
    cur_idx         =   0
    for ii in range(1, len(transitions)):
        # Find transition boundaries
        trans_ii    =   transitions[ii]
        trans_idx   =   np.argmin( abs(np.subtract(range(duration), trans_ii)) )

        # Take a random step
        rand_step   =   np.sign( np.random.normal(0,1,[n_processes,1]) ) * factor

        # update the walk
        rW[:, trans_idx:] = rW[:, trans_idx:] + rand_step

    return rW




