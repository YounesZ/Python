""" This module implements SIMULATION and SAMPLING methods for solving the optimal policy for playing blackjack.
    Simulation:     The state-space is sampled by simulating different games situations and accumulating reward
                    following different strategies (policy)
    Sampling:       The state-space is sampled randomly in a number of blackjack games

    This exercise is part of Sutton and Berto 2003, p.125
    """

from Utils.games.blackjack import *
from random import shuffle, choice
from Utils.programming import ut_ind2sub


class policySolver_blackjack(blackJack):

    def __init__(self, gameType='infinite', method='simulation'):
        # Initialize the engine
        super(policySolver_blackjack, self).__init__(gameType=gameType)
        self.method         =   method
        # Initialize the state values
        self.agent_states   =   list( range(12, 22) )
        self.dealer_states  =   list( range(1, 12))
        self.state_values   =   np.random.random([11, 10])
        # Initialize the policy
        self.dealer_policy  =
        self.agent_policy   =   np.random.random([11, 10])>.5
        self.nEvaluations   =   [0] * 11*10
        self.nUpdates       =   [0] * 11*10
        # Initialize cards pairs
        self.cardMat        =   np.reshape(np.array(range(1, 12)), [11, 1]) + np.array(range(1, 12))
        self.cardVec        =   {1:[1], 2:[2], 3:[3], 4:[4], 5:[5], 6:[6], 7:[7], 8:[8], 9:[9], 10:['Jack', 'Queen', 'King'], 11:[1]}
        self.colors         =   ['Heart', 'Diamond', 'Club', 'Spade']

    def solve_for_state(self, nIterations):
        # ----------
        # --- step1: generate/sample a state-action-reward
        # ----------
        # Select starting state - min nb of updates
        idStart     =   [x if self.nUpdates[x]==min(self.nUpdates) else -1 for x in range(len(self.nUpdates))]
        shuffle(idStart)
        idStart     =   idStart[0]
        # Get states doublet
        y, x        =   ut_ind2sub.main([11, 10], [idStart])
        # Possible cards: dealer
        self.dealer['hand'] =   str(choice(self.cardVec[y[0]+1]))+'_'+choice(self.colors)
        self.hand_value(player='dealer')
        # Possible cards: agent
        iy, ix      =   np.where( self.agent_states[x]==self.cardMat )
        idChoice    =   choice( list(range(len(iy))) )
        self.agent['hand']  =   [str(xlp+1)+'_'+choice(self.colors) for xlp in [iy[idChoice], ix[idChoice]]]
        self.hand_value(player='agent')
        # step2: if the state-action is the current one update its value, else update the policy