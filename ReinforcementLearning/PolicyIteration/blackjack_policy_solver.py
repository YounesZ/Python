""" This module implements SIMULATION and SAMPLING methods for solving the optimal policy for playing blackjack.
    Simulation:     The state-space is sampled by simulating different games situations and accumulating reward
                    following different strategies (policy)
    Sampling:       The state-space is sampled randomly in a number of blackjack games

    This exercise is part of Sutton and Berto 2003, p.125
    """

from Utils.games.blackjack import *


class policySolver_blackjack(blackJack):

    def __init__(self, gameType='infinite', method='simulation'):
        # Initialize the engine
        super(policySolver_blackjack, self).__init__(gameType=gameType)
        self.method         =   method
        # Initialize the state values
        agent_states        =   list( range(12, 22) )
        dealer_states       =   list( range(1, 12))
        self.state_values   =   np.random.random([12, 10])
        # Initialize the policy
        self.policy         =   np.random.random([12, 10])>.5

    def solve_for_state(self, nIterations):

        # step1: generate/sample a state-action-reward

        # step2: if the state-action is the current one update its value, else update the policy