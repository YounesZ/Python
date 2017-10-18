import numpy as np
from random import choice


class blackjack_policy_solver2():

    def __init__(self):
        # States
        self.agent_states   =   list( range(12,22) )
        self.dealer_states  =   list( range(2,12) )
        # Policies
        self.policy_dealer  =   [True]*16 + [False]*5
        self.policy_agent   =   np.random.random([len(self.agent_states), len(self.dealer_states), 2])>0.5

    def init_episode(self):
        # Choose 1 random state for agent
        self.agent  =   {'state':choice(self.agent_states), 'usable':choice([0,1])}
        # Choose 1 random state for dealer
        self.dealer =   {'state':choice(self.dealer_states), 'usable':0}

    def run_episode(self):
        # Init variables
        state_chain     =   [[self.agent['state'], self.dealer['state']]]
        action_chain    =   []
        usable_chain    =   [int(bool(self.agent['usable']))]
        reward          =   0

        # Agent's turn
        loopon  =   True
        action  =   bool(choice([0,1]))
        while loopon:
            # HIT
            if action:
                action_chain.append(True)
                newCard                 =   choice( range(2,12) )
                self.agent['state']     +=  newCard
                if newCard == 11: self.agent['usable'] += 1
                if self.agent['state']>21 and bool(self.agent['usable']):
                    self.agent['state'] -=  10
                    self.agent['usable']-=  1
                elif self.agent['state']>21:
                    reward  =   -1
                    loopon  =   False
                if self.agent['state'] <= 21:
                    action      =   self.policy_agent[self.agent['state']-12, self.dealer['state']-2, self.agent['usable']]
                    state_chain.append([self.agent['state'], self.dealer['state']])
                    usable_chain.append(int(bool(self.agent['usable'])))
            else: # STICK
                action_chain.append(0)
                loopon =    False

        # Dealer's turn
        loopon  =   True
        action  =   self.policy_dealer[self.dealer['state']-1]
        card2   =   choice( range(2,12) )
        evalC   =   True
        self.dealer['usable']   =   int(self.dealer['state']==11) + int(card2==11)
        while loopon and reward>-1:
            # HIT
            if action:
                self.dealer['state']    +=  card2
                if card2 == 11: self.agent['usable'] += 1
                card2 = choice(range(2, 12))
                if self.dealer['state']>21 and bool(self.dealer['usable']):
                    self.dealer['state'] -=  10
                    self.dealer['usable']-=  1
                elif self.dealer['state']>21:
                    reward  =   1
                    loopon  =   False
                    evalC   =   False
            else: #STICK
                loopon  =   False
            # Check game status
            if self.agent['state']==self.dealer['state'] and evalC:
                reward  =   0
            elif self.agent['state'] > self.dealer['state'] and evalC:
                reward  =   1
            elif evalC:
                reward  =   -1
            action      =   self.policy_dealer[min(20,self.dealer['state'])]
        return reward, state_chain, action_chain, usable_chain

    def monte_carlo_ES(self, nIterations):
        # Init variables
        returns     =   np.zeros([len(self.agent_states), len(self.dealer_states), 2, 2])
        visits      =   np.ones([len(self.agent_states), len(self.dealer_states), 2, 2])
        # Loop
        for ii in range(nIterations):
            if not ii%1000: print('Iteration '+str(ii))
            # Init episode
            self.init_episode()
            # Run the episode
            reward, state_chain, action_chain, usable_chain     =   self.run_episode()
            # Distribute reward
            for st, ac, us in zip(state_chain, action_chain, usable_chain):
                returns[st[0]-12, st[1]-2, us, int(ac)]   +=  reward
                visits[st[0]-12, st[1]-2, us, int(ac)]    +=  1
            # State value and policy
            self.action_value   =   returns / visits
            self.policy_agent   =   np.argmax( self.action_value, axis=3 )


# ==== LAUNCHER
BJS2    =   blackjack_policy_solver2()
BJS2.monte_carlo_ES(100000)
