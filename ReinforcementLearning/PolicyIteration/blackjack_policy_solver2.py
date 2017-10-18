import numpy as np
from random import choice, random
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from mpl_toolkits.mplot3d import Axes3D

class blackjack_policy_solver2():

    def __init__(self, initMode='random', gameType='finite'):
        # States
        self.agent_states   =   list( range(12,22) )
        self.dealer_states  =   list( range(2,12) )
        # Policies
        self.policy_dealer  =   [True]*16 + [False]*5
        if initMode=='random':
            self.policy_agent   =   np.random.random([len(self.agent_states), len(self.dealer_states), 2])>0.5
        elif initMode=='stick20':
            self.policy_agent   =   np.concatenate( (np.ones([8, 10, 2]), np.zeros([2, 10, 2])), axis=0 )
        # Card probabilities
        self.gameType       =   gameType
        self.card_reset()

    def card_reset(self):
        self.cardProba      =   [1/13]*8 + [4/13] + [1/13]

    def card_draw(self, idCard=None, exception=None):
        # Set probabilities
        cardP   =   self.cardProba
        if exception is not None:
            cardP[exception-2]  =   -10000
        # Select one card from the deck
        if idCard is None:
            idCard  =   np.argmax( [x+random() for x in cardP] ) + 2
        # Set probabilities
        if self.gameType=='finite':
            self.cardProba[idCard-2]    -=  1/52
        return idCard

    def init_episode(self):
        self.card_reset()
        # Choose 1 random state for agent
        isUsable    =   choice([0,1])
        curState    =   choice(self.agent_states)
        if isUsable:
            card1   =   11
        else:
            card1   =   self.card_draw(exception=11)
        card2       =   curState - card1
        self.agent  =   {'state':curState, 'usable':isUsable}

        # Choose 1 random state for dealer
        self.dealer =   {'state':self.card_draw(choice(self.dealer_states)), 'usable':0}

    def run_episode(self):
        # Init variables
        state_chain     =   [[self.agent['state'], self.dealer['state']]]
        action_chain    =   []
        usable_chain    =   [int(bool(self.agent['usable']))]
        reward          =   0

        # Agent's turn
        loopon  =   True
        action  =   choice([0,1])
        while loopon:
            # HIT
            if action:
                action_chain.append(1)
                newCard                 =   self.card_draw()
                self.agent['state']     +=  newCard
                if newCard == 11: self.agent['usable'] += 1
                if self.agent['state']>21 and bool(self.agent['usable']):
                    self.agent['state'] -=  10
                    self.agent['usable']-=  1
                elif self.agent['state']>21:
                    reward  =   -1
                    loopon  =   False
                if self.agent['state'] <= 21:
                    action      =   deepcopy(self.policy_agent[self.agent['state']-12, self.dealer['state']-2, self.agent['usable']])
                    state_chain.append([self.agent['state'], self.dealer['state']])
                    usable_chain.append(int(bool(self.agent['usable'])))
            else: # STICK
                action_chain.append(0)
                loopon =    False

        # Dealer's turn
        loopon  =   reward>-1
        action  =   self.policy_dealer[self.dealer['state']-1]
        card2   =   self.card_draw()
        evalC   =   True
        self.dealer['usable']   =   int(self.dealer['state']==11) + int(card2==11)
        while loopon:
            # HIT
            if action:
                self.dealer['state']    +=  card2
                if card2 == 11: self.agent['usable'] += 1
                card2 = self.card_draw()
                if self.dealer['state']>21 and bool(self.dealer['usable']):
                    self.dealer['state'] -=  10
                    self.dealer['usable']-=  1
                elif self.dealer['state']>21:
                    reward  =   1
                    loopon  =   False
                    evalC   =   False
                #state_chain.append([self.agent['state'], self.dealer['state']])
            else: #STICK
                loopon  =   False
                # Check game status
                if self.agent['state']==self.dealer['state'] and evalC:
                    reward  =   0
                elif self.agent['state'] > self.dealer['state'] and evalC:
                    reward  =   1
                elif evalC:
                    reward  =   -1
            action      =   deepcopy(self.policy_dealer[min(20,self.dealer['state']-1)])
        return reward, state_chain, action_chain, usable_chain

    def monte_carlo_ES(self, nIterations):
        # Init variables
        returns     =   np.zeros([len(self.agent_states), len(self.dealer_states), 2, 2])
        visits      =   np.ones([len(self.agent_states), len(self.dealer_states), 2, 2])
        # Loop
        for ii in range(nIterations):
            if not ii%5000: print('Iteration '+str(ii))
            # Init episode
            self.init_episode()
            # Run the episode
            reward, state_chain, action_chain, usable_chain     =   self.run_episode()
            #print(state_chain)
            #print('actions', action_chain)
            #print('usable', usable_chain)
            #print('reward', reward)
            # Distribute reward
            for st, ac, us in zip(state_chain, action_chain, usable_chain):
                returns[st[0]-12, st[1]-2, us, ac]  +=  reward
                visits[st[0]-12, st[1]-2, us, ac]   +=  1
            # State value and policy
            self.action_value   =   returns / visits
            self.policy_agent   =   np.argmax( self.action_value, axis=3 )
            self.state_value    =   np.max( self.action_value, axis=3 )

    def print_policy(self):
        fig     =   plt.figure()
        # Axis 1: unusable ace
        Z       =   self.policy_agent[:, :, 0]
        ax1     =   fig.add_subplot(121)
        surf1   =   ax1.imshow(Z)
        ax1.set_xlabel("Dealer's states")
        ax1.set_ylabel("Agent's states")
        ax1.set_title('$\pi_*$: unusable ace')
        ax1.set_xticks(list(range(0, len(self.dealer_states), 2)))
        ax1.set_xticklabels([self.dealer_states[x] for x in range(0, len(self.dealer_states), 2)])
        ax1.set_yticks(list(range(0, len(self.agent_states), 2)))
        ax1.set_yticklabels([self.agent_states[x] for x in range(0, len(self.agent_states), 2)])
        ax1.invert_yaxis()

        # Axis 2: usable ace
        Z       =   self.policy_agent[:, :, 1]
        ax2     =   fig.add_subplot(122)
        surf2   =   ax2.imshow(Z)
        ax2.set_xlabel("Dealer's states")
        ax2.set_ylabel("Agent's states")
        ax2.set_title('$\pi_*$: usable ace')
        ax2.set_xticks(list(range(0, len(self.dealer_states), 2)))
        ax2.set_xticklabels([self.dealer_states[x] for x in range(0, len(self.dealer_states), 2)])
        ax2.set_yticks(list(range(0, len(self.agent_states), 2)))
        ax2.set_yticklabels([self.agent_states[x] for x in range(0, len(self.agent_states), 2)])
        ax2.invert_yaxis()

    def print_stateValue(self):
        fig     =   plt.figure()
        X, Y    =   np.meshgrid(self.agent_states, np.flipud(self.dealer_states))
        # Axis 1: usable ace
        Z       =   np.transpose(np.reshape(self.state_value[:,:,1], [len(self.agent_states), len(self.dealer_states)]))
        ax1     =   fig.add_subplot(121, projection='3d')
        surf1   =   ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax1.set_xlim(self.agent_states[0], self.agent_states[-1])
        ax1.set_ylim(self.dealer_states[0], self.dealer_states[-1])
        ax1.set_zlim([-1, 1])
        ax1.set_xlabel("Agent's states")
        ax1.set_ylabel("Dealer's states")
        ax1.set_zlabel('Value')
        ax1.set_zticks([-1, 0, 1])
        ax1.set_title('Usable ace')
        ax1.invert_yaxis()
        # Axis 2: unusable ace
        Z       =   np.transpose(np.reshape(self.state_value[:,:,0], [len(self.agent_states), len(self.dealer_states)]))
        ax2     =   fig.add_subplot(122, projection='3d')
        surf2   =   ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax2.set_xlim(self.agent_states[0], self.agent_states[-1])
        ax2.set_ylim(self.dealer_states[0], self.dealer_states[-1])
        ax2.set_zlim([-1, 1])
        ax2.set_xlabel("Agent's states")
        ax2.set_ylabel("Dealer's states")
        ax2.set_zlabel('Value')
        ax2.set_zticks([-1, 0, 1])
        ax2.set_title('Unusable ace')
        ax2.invert_yaxis()



# ==== LAUNCHER
BJS    =   blackjack_policy_solver2(initMode='random')
BJS.monte_carlo_ES(500000)
BJS.print_policy()
BJS.print_stateValue()











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