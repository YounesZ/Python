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
        self.state_value    =   np.random.random([len(self.agent_states), len(self.dealer_states), 2])
        # Initialize the action values
        self.action_value   =   np.zeros([len(self.agent_states), len(self.dealer_states), 2, 2])
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
            self.policy_agent   =   np.random.random([len(self.agent_states), len(self.dealer_states), 2]) > .5  # True means HIT, False means STICK
        else:
            # True means HIT, False means STICK
            self.policy_agent   =   policy
        self.nEvaluations       =   np.zeros([len(self.agent_states), len(self.dealer_states)])

    def episode_initialize(self, method='simulation'):
        # Select starting state - min nb of updates
        if method       ==  'simulation':
            # In case we "simulate" the states, we can chose to sample states for which sampling is lowest
            idStart     =   (np.where(self.nEvaluations==np.min(self.nEvaluations)))
            chxSt       =   choice( list(range(len(idStart[0]))) )
            y,x         =   idStart[0][chxSt], idStart[1][chxSt]
        elif method     ==  'exploringStarts':
            y,x         =   [choice(range(len(self.agent_states))), choice(range(len(self.dealer_states)))]
        else:
            # In case we "sample" the states, we simply draw them randomly
            y,x         =   choice(range(len(self.agent_states))), choice(range(len(self.dealer_states)))
        self.current_state  =   [y,x]
        # Set cards: dealer
        dealer      =   {'hand': [str(choice(self.cardVec[self.dealer_states[x]])) + '_' + choice(self.colors), str(choice(self.cardVec[choice(range(2,12))])) + '_' + choice(self.colors)], 'shown': [True, False], 'plays': [], 'value': 0, 'status': 'On', 'usable': False}
        # Set cards: agent
        iy, ix      =   np.where(self.agent_states[y] == self.cardMat)
        iy          +=  1
        ix          +=  1
        idUsable    =   choice( np.where([iy[lp]==11 or ix[lp]==11 for lp in range(len(iy))])[0] )
        idUnusable  =   np.where([iy[lp]!=11 and ix[lp]!=11 for lp in range(len(iy))])[0]
        if bool(choice([0,1])) or not len(idUnusable):
            # Usable - need an ace
            agent       =   {'hand': [str(min(xlp%11+1, xlp)) + '_' + choice(self.colors) for xlp in [iy[idUsable], ix[idUsable]]], 'shown': [True, True], 'plays': [], 'value': 0, 'status': 'On', 'usable': False}
        else: # Unusable ace
            idUnusable  =   choice(idUnusable)
            agent       =   {'hand': [str(min(xlp%11+1, xlp)) + '_' + choice(self.colors) for xlp in [iy[idUnusable], ix[idUnusable]]], 'shown': [True, True], 'plays': [], 'value': 0, 'status': 'On', 'usable': False}
        # Init game
        self.deck_new(statusOnly=True, printStatus=False, initCards=[agent, dealer])

    def episode_run(self, randomInit=True):
        # ----------
        # play the game
        self.turn       =   'agent'
        curState        =   self.current_state
        reward          =   2
        state_chain     =   [curState]
        usable_chain    =   [self.agent['usable']]
        action_chain    =   []
        while self.turn == 'agent' and reward == 2:
            # --- Pick action according to agent's policy
            # Pick agent action at that state according to policy
            if randomInit:
                action      =   bool( choice([0,1]) )
                randomInit  =   False
            else:
                action      =   self.policy_agent[curState[0], curState[1], int(self.agent['usable'])]
            if action:
                self.hand_do('hit', statUpd=False)
                action_chain.append(1)
                if self.agent['value']>0:
                    # Determine in which state we are now
                    curState = [self.agent_states.index(self.agent['value']), self.dealer_states.index(self.dealer['value'])]
                    # Append it to state chain
                    state_chain.append(curState)
                    usable_chain.append(self.agent['usable'])
            else:  # Stick
                self.hand_do('stick', statUpd=False)
            reward = self.game_status(statusOnly=True, printStatus=False)
        action_chain.append(0)
        while self.turn == 'dealer' and reward == 2:
            # --- Pick action according to dealer's policy
            if self.policy_dealer[self.dealer['value'] - 1]:  # Hit
                self.hand_do('hit', statUpd=False)
            else:
                self.hand_do('stick', statUpd=False)
            reward = self.game_status(statusOnly=True, printStatus=False)
        return reward, state_chain, usable_chain, action_chain

    def evaluate_policy(self, nIterations=10000):
        # Initialize the algorithm
        returns =   np.random.random([len(self.agent_states), len(self.dealer_states), 2])
        visits  =   np.random.random([len(self.agent_states), len(self.dealer_states), 2])
        # Start looping
        for ii in range(nIterations):
            # --- step1: initialize the episode
            self.episode_initialize()
            # --- step2: run the episode
            reward, state_chain, usable_chain, _    =   self.episode_run()
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
        returns     =   np.zeros(np.shape(self.action_value))
        visits      =   np.ones(np.shape(self.action_value))
        # Start looping
        for ii in range(nIterations):
            if not ii%1000: print('Iteration'+str(ii))
            # --- step1: initialize the episode
            self.episode_initialize(method='exploringStarts')
            # --- step2: run the episode
            reward, state_chain, usable_chain, action_chain     =   self.episode_run(randomInit=True)
            #self.game_status(statusOnly=True)
            #print(state_chain)
            #print(action_chain)
            self.nEvaluations[self.current_state] += 1
            self.history.append(reward)
            # Unfold the reward onto chain
            firstVisit  =   dict((tuple(el),0) for el in state_chain)
            for st, ac, us in zip(state_chain, action_chain, usable_chain):
                if not firstVisit[tuple(st)]:
                    returns[st[0], st[1], int(us), ac]  +=  [reward]
                    visits[st[0], st[1], int(us), ac]   +=  1
                    firstVisit[tuple(st)]               +=  1
            # Store new state values
            self.action_value   =   returns /   visits
            # Set new policy
            self.policy_agent   =   np.argmax( self.action_value, axis=3 )



# ========
# LAUNCHER
# ========
# Instantiate the solver
BJS     =   policySolver_blackjack(method='sampling', gameType='finite')
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
BJS.solve_policy_MC(nIterations=500000)


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
# Axis 1: unusable ace
Z       =   BJS.policy_agent[:,:,0]
ax1     =   fig.add_subplot(121)
surf1   =   ax1.imshow( Z )
ax1.set_xlabel("Dealer's states")
ax1.set_ylabel("Agent's states")
ax1.set_title('$\pi_*$: unusable ace')
ax1.set_xticks(list(range(0,len(BJS.dealer_states),2)))
ax1.set_xticklabels([BJS.dealer_states[x] for x in range(0,len(BJS.dealer_states),2)])
ax1.set_yticks(list(range(0,len(BJS.agent_states),2)))
ax1.set_yticklabels([BJS.agent_states[x] for x in range(0,len(BJS.agent_states),2)])
ax1.invert_yaxis()

# Axis 2: usable ace
Z       =   BJS.policy_agent[:,:,1]   
ax2     =   fig.add_subplot(122)
surf2   =   ax2.imshow(Z)
ax2.set_xlabel("Dealer's states")
ax2.set_ylabel("Agent's states")
ax2.set_title('$\pi_*$: usable ace')
ax2.set_xticks(list(range(0,len(BJS.dealer_states),2)))
ax2.set_xticklabels([BJS.dealer_states[x] for x in range(0,len(BJS.dealer_states),2)])
ax2.set_yticks(list(range(0,len(BJS.agent_states),2)))
ax2.set_yticklabels([BJS.agent_states[x] for x in range(0,len(BJS.agent_states),2)])
ax2.invert_yaxis()




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