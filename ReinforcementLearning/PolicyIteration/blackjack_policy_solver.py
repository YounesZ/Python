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
        # Initialize the action values
        self.action_value_usable    =   np.zeros([self.nStates,2])
        self.action_value_unusable  =   np.zeros([self.nStates, 2])
        # Initialize cards pairs
        self.cardMat        =   np.reshape(np.array(range(1, 12)), [11, 1]) + np.array(range(1, 12))
        self.cardVec        =   {2:[2], 3:[3], 4:[4], 5:[5], 6:[6], 7:[7], 8:[8], 9:[9], 10:['jack', 'queen', 'king'], 11:[1]}
        self.colors         =   ['Heart', 'Diamond', 'Club', 'Spade']

    def set_policy(self, policy=None):
        # Dealer policy: hard-coded - dealer sticks on 17 and higher
        self.policy_dealer  =   np.array([1] * 16 + [0] * 5) > 0  # True means HIT, False means STICK
        # Agent policy
        if policy is None:
            # Initialize a random one
            self.policy_usable  =   np.random.random(self.nStates) > .5  # True means HIT, False means STICK
            self.policy_unusable=   np.random.random(self.nStates) > .5
        else:
            # True means HIT, False means STICK
            self.policy_usable  =   policy[0]
            self.policy_unusable=   policy[1]
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
        # Set cards: dealer
        dealer      =   {'hand': [str(choice(self.cardVec[self.dealer_states[y[0]]])) + '_' + choice(self.colors), str(choice(self.cardVec[choice(range(2,12))])) + '_' + choice(self.colors)], 'shown': [True, False], 'plays': [], 'value': 0, 'status': 'On', 'usable': False}
        # Set cards: agent
        iy, ix      =   np.where(self.agent_states[x[0]] == self.cardMat)
        idChoice    =   choice(list(range(len(iy))))
        agent       =   {'hand': [str(xlp%10 + 1) + '_' + choice(self.colors) for xlp in [iy[idChoice], ix[idChoice]]], 'shown': [True, True], 'plays': [], 'value': 0, 'status': 'On', 'usable': False}
        # Init game
        self.game_start(statusOnly=True, printStatus=False, initCards=[agent, dealer])

    def episode_run(self):
        # ----------
        # play the game
        self.turn       =   'agent'
        curState        =   self.agent_states.index(self.agent['value']) * len(self.dealer_states) + self.dealer_states.index(self.dealer['value'])
        print(self.current_state, curState)
        reward          =   2
        state_chain     =   [curState]
        usable_chain    =   [self.agent['usable']]
        while self.turn == 'agent' and reward == 2:
            # --- Pick action according to agent's policy
            # Pick agent action at that state according to policy
            action      =   'stick'
            if self.agent['usable'] and self.policy_usable[curState]:  # Hit
                action  =   'hit'
            elif not self.agent['usable'] and self.policy_unusable[curState]:
                action  =   'hit'
            if action   ==  'hit':
                self.hand_do('hit', statUpd=False)
                if self.agent['value']>0:
                    # Determine in which state we are now
                    curState = self.agent_states.index(self.agent['value']) * len(self.dealer_states) + self.dealer_states.index(self.dealer['value'])
                    # Append it to state chain
                    state_chain.append(curState)
                    usable_chain.append(self.agent['usable'])
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

    def solve_policy_MC(self, nIterations=10000):
        # Initialize the algorithm
        returns_usable      =   [[[] for _ in range(self.nStates)] for _ in range(2)]
        returns_unusable    =   [[[] for _ in range(self.nStates)] for _ in range(2)]
        # Start looping
        for ii in range(nIterations):
            # --- step1: initialize the episode
            self.episode_initialize()
            # --- step2: run the episode
            reward, state_chain, usable_chain = self.episode_run()
            self.nEvaluations[self.current_state] += 1
            self.history.append(reward)
            # Unfold the reward onto chain
            action_chain            =   [1]*(len(state_chain)-1) + [0]
            #action_chain            =   [int(self.policy_agent[0,x]) for x in state_chain]
            for st, ac, us in zip(state_chain, action_chain, usable_chain):
                if us:
                    returns_usable[ac][st]  +=   [reward]
                else:
                    returns_unusable[ac][st]+=   [reward]
            # Store new state values
            self.action_value_usable    =   [[np.mean(y) if len(y)>0 else (np.random.random()-.5)/10 for y in x] for x in returns_usable]
            self.action_value_unusable  =   [[np.mean(y) if len(y)>0 else (np.random.random()-.5)/10 for y in x] for x in returns_unusable]
            self.policy_usable          =   [self.action_value_usable[1][x] > self.action_value_usable[0][x] for x in range(self.nStates)]
            self.policy_unusable        =   [self.action_value_unusable[1][x] > self.action_value_unusable[0][x] for x in range(self.nStates)]


# ========
# LAUNCHER
# ========
# Instantiate the solver
BJS     =   policySolver_blackjack(method='sampling', gameType='infinite')
# -----POLICY EVALUATION
# Make the agent's policy
#policyAG=   np.reshape([x<20 for x in BJS.agent_states], [len(BJS.agent_states),1]) * [x>0 for x in BJS.dealer_states]
#policyAG=   np.reshape(policyAG, [1, BJS.nStates])
#BJS.set_policy(policyAG)
# Evaluate that policy
#BJS.evaluate_policy(nIterations=10000)
# -----POLICY ITERATION
# Solve for policy - Monte-Carlo algorithm
BJS.set_policy()
BJS.solve_policy_MC(nIterations=10000)


"""

# ========
# DISPLAY
# ========
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from mpl_toolkits.mplot3d import Axes3D


# --- POLICY ---
fig     =   plt.figure()
# Axis 1: usable ace
Z       =   np.reshape(BJS.policy_usable, [len(BJS.agent_states), len(BJS.dealer_states)])
ax1     =   fig.add_subplot(121)
surf1   =   ax1.imshow( Z )
ax1.set_xlabel("Dealer's states")
ax1.set_ylabel("Agent's states")
ax1.set_title('$\pi_*$: usable ace')
ax1.set_xticks(list(range(0,len(BJS.dealer_states),2)))
ax1.set_xticklabels([BJS.dealer_states[x] for x in range(0,len(BJS.dealer_states),2)])
ax1.set_yticks(list(range(0,len(BJS.agent_states),2)))
ax1.set_yticklabels([BJS.agent_states[x] for x in range(0,len(BJS.agent_states),2)])
# Axis 2: unusable ace
Z       =   np.reshape(BJS.policy_unusable, [len(BJS.agent_states), len(BJS.dealer_states)])
ax2     =   fig.add_subplot(122)
surf2   =   ax2.imshow(Z)
ax2.set_xlabel("Dealer's states")
ax2.set_ylabel("Agent's states")
ax2.set_title('$\pi_*$: unusable ace')
ax2.set_xticks(list(range(0,len(BJS.dealer_states),2)))
ax2.set_xticklabels([BJS.dealer_states[x] for x in range(0,len(BJS.dealer_states),2)])
ax2.set_yticks(list(range(0,len(BJS.agent_states),2)))
ax2.set_yticklabels([BJS.agent_states[x] for x in range(0,len(BJS.agent_states),2)])




# --- STATE VALUE ---
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
"""