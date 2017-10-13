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
        self.nStates        =   11*10
        self.state_value    =   np.random.random([1, self.nStates])
        # Initialize cards pairs
        self.cardMat        =   np.reshape(np.array(range(1, 12)), [11, 1]) + np.array(range(1, 12))
        self.cardVec        =   {1:[1], 2:[2], 3:[3], 4:[4], 5:[5], 6:[6], 7:[7], 8:[8], 9:[9], 10:['jack', 'queen', 'king'], 11:[1]}
        self.colors         =   ['Heart', 'Diamond', 'Club', 'Spade']

    def set_policy(self, policy=None):
        # Dealer policy: hard-coded
        self.policy_dealer  =   np.array([1] * 17 + [0] * 4) > 0  # True means HIT, False means STICK
        # Agent policy
        self.policy_agent   =   policy
        if policy=='None':
            # Initialize a random one
            self.policy_agent=   np.random.random([1, self.nStates]) > .5  # True means HIT, False means STICK
        self.nEvaluations   =   [0] * self.nStates
        self.nUpdates       =   [0] * self.nStates

    def evaluate_policy(self, nIterations=10000):
        # Initialize the algorithm
        value       =   deepcopy(self.state_value)
        returns     =   [[]] * self.nStates
        # Start looping
        for ii in range(nIterations):
            # ----------
            # --- step1: initialize the exploration state
            # ----------
            # Select starting state - min nb of updates
            if self.method=='simulation':
                # In case we "simulate" the states, we can chose to sample states for which sampling is lowest
                idStart =   [x if self.nUpdates[x]==min(self.nUpdates) else -1 for x in range(self.nStates)]
            else:
                # In case we "sample" the states, we simply draw them randomly
                idStart =   list( range(len(self.nUpdates)) )
            shuffle(idStart)
            idStart     =   idStart[0]
            # Get states doublet
            y, x        =   ut_ind2sub.main([11, 10], [idStart])
            # Possible cards: dealer
            self.game_start(statusOnly=True, printStatus=False)
            self.dealer['hand'][0]=   str(choice(self.cardVec[y[0]+1]))+'_'+choice(self.colors)
            self.hand_value(player='dealer')
            # Possible cards: agent
            iy, ix      =   np.where( self.agent_states[x[0]]==self.cardMat )
            idChoice    =   choice( list(range(len(iy))) )
            self.agent['hand']  =   [str(xlp+1)+'_'+choice(self.colors) for xlp in [iy[idChoice], ix[idChoice]]]
            self.hand_value(player='agent')
            self.game_status(statusOnly=True)
            # ----------
            # --- step2: generate/sample a state-action-reward
            # ----------
            # play the game
            reward          =   2
            state_chain     =   []
            while self.turn=='agent' and reward==2:
                # --- Pick action according to agent's policy
                # Determine in which state we are
                curState    =   self.agent_states.index(self.agent['value'])*len(self.dealer_states) + self.dealer_states.index(self.dealer['value'])
                # Append it to state chain
                state_chain.append(curState)
                # Pick agent action at that state according to policy
                if self.policy_agent[0,curState]: # Hit
                    self.hand_do('hit', statUpd=False)
                else: # Stick
                    self.hand_do('stick', statUpd=False)
                reward  =   self.game_status(statusOnly=True)
            while self.turn=='dealer' and reward==2:
                # --- Pick action according to dealer's policy
                if self.policy_dealer[self.dealer['value']-1]: # Hit
                    self.hand_do('hit', statUpd=False)
                else:
                    self.hand_do('stick', statUpd=False)
                reward  =   self.game_status(statusOnly=True)
            self.history.append(reward)
            # Unfold the reward onto chain
            returns     =   [returns[x]+[reward] if x in state_chain else returns[x] for x in range(self.nStates)]
            # Update policy
            value       =   [np.mean(x) for x in returns]
        # Store new state values
        self.state_value=   value



# ========
# LAUNCHER
# ========
# Instantiate the solver
#BJS     =   policySolver_blackjack()
# Make the agent's policy
#policyAG=   np.reshape([x<20 for x in BJS.agent_states], [len(BJS.agent_states),1]) * [x>0 for x in BJS.dealer_states]
#policyAG=   np.reshape(policyAG, [1, BJS.nStates])
#BJS.set_policy( policyAG)
# Evaluate that policy
#BJS.evaluate_policy()
