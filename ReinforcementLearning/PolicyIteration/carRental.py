""" This is a first attempt to implement policy iteration strategy to solve Jack's car rental problem 4.5 found in
Sutton and Barto 2003 p.93

This first version considers two car rental agencies that both cumulate 20 cars. Each rented car makes a 10$/day profit
and overnight car transfers between the two agencies cost 2$ per car. We are trying to find the optimal policy in terms
of total reward

Expected cars rented in agencies 1 and 2 are respectively Poisson random variables with lambda=3 and 4
Expected cars returned to agencies 1 and 2 are respectively Poisson random variables with lambda=3 and 2

"""

import numpy as np
import matplotlib.pyplot as plt
import math


def main():

    # Problem settings
    total_cars      =   20      # among the 2 agencies\
    rental_profit   =   10      # per car, per day
    max_transfer    =   5       # par night
    transfer_cost   =   2       # per car
    expected_rent   =   [3, 4]
    expected_return =   [3, 2]
    theta           =   0.1     # change tolerance, useful in policy evaluation
    stable_policy   =   False

    # Initialization
    state_value     =   np.zeros([total_cars, total_cars])
    optimal_policy  =   np.zeros([total_cars, total_cars])


    # ===============
    #
    # Helper functions
    #
    # ===============

    def poisson_proba(mylambda, values):
        proba   =   np.zeros([1, len(values)])
        for x in range( len(values) ):
            lambda_n    =   mylambda ** values[x]
            n_fact      =   math.factorial(values[x])
            proba[0,x]  =   lambda_n / n_fact * math.exp(-mylambda)
        return proba

    def update_value(s):
        # Convert state to nb of cars in each agency
        agency1_cars=   s%total_cars
        agency2_cars=   int(np.floor(s/total_cars))

        # Add vehicles moved according to pi
        moved       =   optimal_policy[agency1_cars, agency2_cars]

        # Probability of rents and returns - agency1
        proba1_rent =   poisson_proba(expected_rent[0], range(total_cars))
        proba1_rtrn =   poisson_proba(expected_return[0], range(total_cars))
        jointProba1 =   np.reshape(proba1_rent, [total_cars, 1]) * proba1_rtrn

        proba2_rent = poisson_proba(expected_rent[1], range(total_cars))
        proba2_rtrn =   poisson_proba(expected_return[1], range(total_cars))


        # Probability vector
        current_cars1   =   agency1_cars + moved - range(total_cars)
        current_cars2   =   agency2_cars - moved - range(total_cars)
        proba1[0, agency1_cars:]    =   0
        proba2[0, agency2_cars:]    =   0
        jointProba  =   np.reshape(proba1) * proba2



        proba_out   =   .po expected_rent
        return state_value

    def evaluate_policy():
        # intialize change
        value_change = 0
        for sii in range(len(state_value)):
            temp = state_value[sii]

    def improve_policy():

    def compare_policy():



    # Solve MDP
    while ~stable_policy:
        state_value     =   evaluate_policy()
        new_policy      =   improve_policy()
        stable_policy   =   compare_policy()
        optimal_policy  =   new_policy

    return optimal_policy


# Driver
if __name__ == "__main__":
    main()