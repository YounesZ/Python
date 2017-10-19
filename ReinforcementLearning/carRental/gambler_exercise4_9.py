""" This is a first attempt to implement policy iteration strategy to solve Jack's car rental problem 4.5 found in
Sutton and Barto 2003 p.93

This second version considers two car rental agencies that both cumulate 20 cars. Each rented car makes a 10$/day profit
and overnight car transfers between the two agencies cost 2$ per car. We are trying to find the optimal policy in terms
of total reward.
*** Changes with respect to version 1: Each agency now gets a negative reward to compensate for the "manque a gagner"
    when a customer walks in to rent a car but does not find any available

Expected cars rented in agencies 1 and 2 are respectively Poisson random variables with lambda=3 and 4
Expected cars returned to agencies 1 and 2 are respectively Poisson random variables with lambda=3 and 2

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from Utils.programming import ut_sum_diagonals, ut_closest, ut_ind2sub
from Utils.maths import ut_poisson_proba
from random import shuffle
import time


# ===============
# Helper functions
# ===============
def update_value(stake, state_value, state, params):
    # At each play, two states can be reached
    stateP1     =   state + stake
    stateP2     =   state - stake

    # Probability of each state
    probaP1     =   params.get('p_head')
    probaP2     =   1 - params.get('p_head')

    # Values of next states
    valueN1     =   state_value[0, int(stateP1)]
    valueN2     =   state_value[0, int(stateP1)]

    # State rewards
    rew1        =   0
    rew2        =   0
    if stateP1 ==   params.get('win_state'):
        rew1    =   1
    # Issue state value
    v = probaP1 * (rew1 + valueN1*params.get('gamma')) + probaP2 * (rew2 + valueN2*params.get('gamma'))
    return v


def evaluate_policy(optimal_policy, state_value, params):
    # intialize change
    n_passes    =   0
    delta_v     =   float('Inf')
    while delta_v > params.get('theta'):
        delta_v = 0
        # Shuffle the order of value evaluation
        shufI   =   np.array( range(params.get('win_state')-1) ) + 1
        #shuffle(shufI)
        for sii in shufI:
            temp    =   state_value[0,sii]
            new_v   =   update_value(optimal_policy[0,sii], state_value, sii, params)
            state_value[0,sii]    =   new_v
            delta_v =   np.maximum(delta_v, np.abs(temp - new_v))
        n_passes += 1
    return state_value, n_passes


def improve_policy(optimal_policy, state_value, params):
    policy_stable   =   True
    n_states        =   np.array( range(params.get('win_state')-1) ) + 1
    #shuffle(n_states)
    # Predefined actions
    for sii in n_states:
        # Possible stakes from current state
        actions_poss=   np.array(range(min(sii, params.get('win_state')-sii)+1))
        temp        =   optimal_policy[0,sii]
        # Loop on all possible actions and recompute the value
        for sjj in actions_poss:
            new_val =   update_value(sjj, state_value, sii, params)
            if new_val>state_value[0,sii]:
                state_value[0,sii]      =   new_val
                optimal_policy[0,sii]   =   sjj
        if temp!=optimal_policy[0,sii]:
            policy_stable   =   False
    return optimal_policy, policy_stable






# ===============
# Main function
# ===============
def main(p_head=0.55, theta=0.1):

    print("\n\n\t***\tStarted to solve 2D MDP with value iteration: the gambler's problem\n\n")

    start_time  =   time.time()

    # Problem settings
    params  =   {'win_state'    : 100,      # among the 2 agencies\
                'loose_state'   : 0,        # per car, per day
                'win_reward'    : 1,        # reward for winning the game
                'loose_reward'  : 0,        # reward for losing the game
                'play_cost'     : 0,        # per trial
                'theta'         : theta,    # change tolerance, useful in policy evaluation
                'gamma'         : 0.9,
                'p_head'        : p_head}   # probability of head side of the coin

    # Initialization
    state_value     =   np.zeros([1, params.get('win_state')+1])
    optimal_policy  =   np.zeros([1, params.get('win_state')+1])

    # Solve MDP - 1 pass only
    state_value, _                  =   evaluate_policy(optimal_policy, state_value, params)
    optimal_policy, stable_policy   =   improve_policy(optimal_policy, state_value, params)

    print("\t--- All done in %s seconds ---" % (time.time() - start_time))

    return state_value, optimal_policy, params



# Driver
if __name__ == "__main__":
    main()