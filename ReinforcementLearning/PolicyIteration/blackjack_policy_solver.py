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
        self.dealer_states  =   list( range(2, 12))
        self.nStates        =   len(self.agent_states)*len(self.dealer_states)
        self.state_value_usable     =   np.random.random([1, self.nStates])
        self.state_value_unusable   =   np.random.random([1, self.nStates])
        # Initialize cards pairs
        self.cardMat        =   np.reshape(np.array(range(1, 12)), [11, 1]) + np.array(range(1, 12))
        self.cardVec        =   {2:[2], 3:[3], 4:[4], 5:[5], 6:[6], 7:[7], 8:[8], 9:[9], 10:['jack', 'queen', 'king'], 11:[1]}
        self.colors         =   ['Heart', 'Diamond', 'Club', 'Spade']

    def set_policy(self, policy=None):
        # Dealer policy: hard-coded - dealer sticks on 17 and higher
        self.policy_dealer  =   np.array([1] * 16 + [0] * 5) > 0  # True means HIT, False means STICK
        # Agent policy
        self.policy_agent   =   policy
        if policy[0][0]=='None':
            # Initialize a random one
            self.policy_agent=   np.random.random([1, self.nStates]) > .5  # True means HIT, False means STICK
        self.nEvaluations   =   [0] * self.nStates
        self.nUpdates       =   [0] * self.nStates

    def episode_initialize(self):
        # Select starting state - min nb of updates
        if self.method == 'simulation':
            # In case we "simulate" the states, we can chose to sample states for which sampling is lowest
            idStart         =   [x if self.nEvaluations[x] == min(self.nEvaluations) else -1 for x in range(self.nStates)]
            idStart         =   ut_remove_value.main(idStart, '>-1')
        else:
            # In case we "sample" the states, we simply draw them randomly
            idStart         =   list(range(self.nStates))
        shuffle(idStart)
        self.current_state  =   idStart[0]
        # Get states doublet
        y, x                =   ut_ind2sub.main([10, 10], [self.current_state])
        # Possible cards: dealer
        self.game_start(statusOnly=True, printStatus=False)
        self.dealer['hand'][0]  =   str(choice(self.cardVec[self.dealer_states[y[0]]])) + '_' + choice(self.colors)
        self.hand_value(player='dealer')
        # Possible cards: agent
        iy, ix              =   np.where(self.agent_states[x[0]] == self.cardMat)
        idChoice            =   choice(list(range(len(iy))))
        self.agent['hand']  =   [str(xlp + 1) + '_' + choice(self.colors) for xlp in [iy[idChoice], ix[idChoice]]]
        self.hand_value(player='agent')

    def episode_run(self):
        # ----------
        # play the game
        reward = 2
        state_chain = []
        usable_chain = []
        while self.turn == 'agent' and reward == 2:
            # --- Pick action according to agent's policy
            # Determine in which state we are
            curState = self.agent_states.index(self.agent['value']) * len(
                self.dealer_states) + self.dealer_states.index(self.dealer['value'])
            # Append it to state chain
            state_chain.append(curState)
            usable_chain.append(self.agent['usable'])
            # Pick agent action at that state according to policy
            if self.policy_agent[0, curState]:  # Hit
                self.hand_do('hit', statUpd=False)
            else:  # Stick
                self.hand_do('stick', statUpd=False)
            reward = self.game_status(statusOnly=True, printStatus=False)
        while self.turn == 'dealer' and reward == 2:
            # --- Pick action according to dealer's policy
            if self.policy_dealer[self.dealer['value'] - 1]:  # Hit
                self.hand_do('hit', statUpd=False)
            else:
                self.hand_do('stick', statUpd=False)
            reward = self.game_status(statusOnly=True, printStatus=False)
        return reward, state_chain, usable_chain

    def evaluate_policy(self, nIterations=10000):
        # Initialize the algorithm
        returns_usable  =   [[]] * self.nStates
        returns_unusable=   [[]] * self.nStates
        # Start looping
        for ii in range(nIterations):
            # --- step1: initialize the episode
            self.episode_initialize()
            # --- step2: run the episode
            reward, state_chain, usable_chain       =   self.episode_run()
            self.nEvaluations[self.current_state]   +=  1
            self.history.append(reward)
            # Unfold the reward onto chain
            returns_usable          =   [returns_usable[x] + [reward] if x in state_chain and usable_chain[state_chain.index(x)] else returns_usable[x] for x in range(self.nStates)]
            returns_unusable        =   [returns_unusable[x] + [reward] if x in state_chain and not usable_chain[state_chain.index(x)] else returns_unusable[x] for x in range(self.nStates)]
        # Store new state values
        self.state_value_usable     =   [np.mean(x) for x in returns_usable]
        self.state_value_unusable   =   [np.mean(x) for x in returns_unusable]



# ========
# LAUNCHER
# ========
# Instantiate the solver
BJS     =   policySolver_blackjack(method='sampling', initial_policy='su')
# Make the agent's policy
policyAG=   np.reshape([x<20 for x in BJS.agent_states], [len(BJS.agent_states),1]) * [x>0 for x in BJS.dealer_states]
policyAG=   np.reshape(policyAG, [1, BJS.nStates])
BJS.set_policy( policyAG)
# Evaluate that policy
BJS.evaluate_policy(nIterations=10000)


# ========
# DISPLAY
# ========
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from mpl_toolkits.mplot3d import Axes3D
fig     =   plt.figure()
X, Y    =   np.meshgrid(BJS.agent_states, np.flipud(BJS.dealer_states))
# Axis 1: usable ace
Z       =   np.transpose(np.reshape(BJS.state_value_usable, [len(BJS.agent_states), len(BJS.dealer_states)]))
ax1     =   fig.add_subplot(121, projection='3d')
surf1   =   ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.set_xlim([BJS.agent_states[0], BJS.agent_states[-1]])
ax1.set_ylim([BJS.dealer_states[0], BJS.dealer_states[-1]])
ax1.set_zlim([-1, 1])
ax1.set_xlabel("Agent's states")
ax1.set_ylabel("Dealer's states")
ax1.set_zlabel('Value')
ax1.set_zticks([-1, 0, 1])
ax1.set_title('Usable ace')
ax1.invert_yaxis()
# Axis 2: unusable ace
Z       =   np.transpose(np.reshape(BJS.state_value_unusable, [len(BJS.agent_states), len(BJS.dealer_states)]))
ax2     =   fig.add_subplot(122, projection='3d')
surf2   =   ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.set_xlim([BJS.agent_states[0], BJS.agent_states[-1]])
ax2.set_ylim([BJS.dealer_states[0], BJS.dealer_states[-1]])
ax2.set_zlim([-1, 1])
ax2.set_xlabel("Agent's states")
ax2.set_ylabel("Dealer's states")
ax2.set_zlabel('Value')
ax2.set_zticks([-1, 0, 1])
ax2.set_title('Unusable ace')
ax2.invert_yaxis()

