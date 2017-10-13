"""
    This module implements blackjack game rules and two different methods for finding the best policy:
    - method1: MC policy evaluation with episode simulation
    - method2: MC policy evaluation with episode averaging

    An episode is a game of blackjack that starts with the dealer giving two cards to the agent/player and two for
    himself (one face up, one face down) and ends when either agent busts, dealer busts or when both of them stick and
    compare their hands for determining win (+1 reward), lose (-1 reward) or draw (0 reward) outcomes

    Two types of games can be played:
    - type1: finite card deck (without replacement, dealer picks a new deck after depletion)
    - type2: infinite card deck (with replacement, dealer picks a new deck after each episode)

    This exercise is drawn from the book of Sutton and Berto (2002) p.108

    TODO:
        -   Implement infinite card deck
        -   Monte-Carlo method for finding optimal policy by simulating state-action
        -   Monte-Carlo method for finding optimal policy by sampling state-action
"""

#==== IMPORTS
import numpy as np
from random import shuffle
from Utils.programming import ut_remove_value
from copy import deepcopy


class blackJack:

    def __init__(self, gameType='infinite'):
        # Data location
        self.gameType   =   gameType
        # Initiate data structures
        self.replace=   {'jack':'10', 'queen':'10', 'king':'10'}
        self.history=   []
        # Initiate game deck
        self.deck_new()

    def deck_new(self):
        # Make a new card deck
        colors              =   ['Heart', 'Diamond', 'Club', 'Spade']
        numbers             =   [str(x + 1) for x in range(10)] + ['jack', 'queen', 'king']
        self.deck           =   list( np.concatenate([[x+'_'+y for x in numbers] for y in colors][:]) )
        self.deck_empty     =   False
        self.cardP          =   [1/52]*52
        #self.game_start()

    def game_start(self, statusOnly=False, printStatus=True):
        self.agent  =   {'hand': [], 'shown': [], 'plays': [], 'value': 0, 'status': 'On'}
        self.dealer =   {'hand': [], 'shown': [], 'plays': [], 'value': 0, 'status': 'On'}
        # Dealer gives two cards to agent
        self.hand_do('hit','agent',False)
        self.hand_do('hit','agent',False)
        self.hand_value(player='agent')
        # Dealer gives two cards to himself
        self.hand_do('hit','dealer',False)
        self.hand_do('hit','dealer',False)
        self.dealer['shown'] = [True, False]
        self.hand_value(player='dealer')
        if self.deck_empty:
            print('\n\tDeck empty, restarting, ...\n\n')
            self.deck_new()
        else:
            # Evaluate new hand value
            self.turn   =   'agent'
            # Evaluate game status
            if printStatus: self.status_print('New game', 2)
            self.game_status(statusOnly=statusOnly, printStatus=printStatus)

    def hand_do(self, action, player=None, statUpd=True):
        if player   ==  None:
            player  =   self.turn
        plDict      =   getattr(self, player)

        if action=='hit':
            if all(x==0 for x in self.cardP):
                self.deck_empty     =   True
            else:
                if player=='dealer' and len(self.dealer['shown'])>0 and not self.dealer['shown'][-1]:
                    self.dealer['shown'][-1]    =   True
                else:
                    # Select card
                    pickP   =   list( np.multiply(self.cardP, abs(np.random.random(52))) )
                    pickIx  =   pickP.index(max(pickP))
                    plDict['hand'].append(self.deck[pickIx])
                    plDict['shown'].append(True)
                    if self.gameType=='finite':
                        self.cardP[pickIx]  =   0
                    setattr(self, player, plDict)
                # Evaluate new hand value
                self.hand_value(player=player)
        elif action=='stick':
            plDict['status']    =   'stick'
            setattr(self, player, plDict)
        if statUpd:
            self.game_status()

    def hand_value(self, player=None):
        # Evaluate player's hand
        if player==None:
            player      =   self.turn
        plDict          =   getattr(self, player)
        # Make sure all cards are converted to value
        cards           =   [x.split('_')[0] if y else '-1' for x,y in zip(plDict['hand'], plDict['shown'])]
        cards           =   [self.replace[x] if x in list(self.replace.keys()) else x for x in cards]
        cards           =   [int(x) for x in ut_remove_value.main(cards, "!=-1")]
        plDict['plays'] =   [deepcopy(cards)]
        # Check for usable aces
        if 1 in cards:
            idx         =   cards.index(1)
            cards[idx]  =   11
            plDict['plays'] +=  [deepcopy(cards)]
        # Compute values
        plDict['value'] =   [sum(x) for x in plDict['plays']]
        # Check if busting
        bust            =   [x>21 for x in plDict['value']]
        bkjack          =   [x==21 for x in plDict['value']]
        # Keep highest value
        highB           =   [x if y==False else 0 for x,y in zip(plDict['value'], bust)]
        plDict['value'] =   max(highB)
        if any(bust):
            plDict['usable']=   False
        if all(bust):
            plDict['status']=   'bust'
        if any(bkjack):
            plDict['status']=   'blackjack'
        setattr(self, player, plDict)

    def game_status(self, statusOnly=False, printStatus=True):
        # Check agent's hand
        oldTurn         =   self.turn
        if self.agent['status'] == 'bust':
            msg         =   'agent loses, busted'
            status      =   -1
            self.turn   =   'dealer'
        elif any(self.agent['status']==x for x in ['blackjack', 'stick']) and self.turn=='agent':
            msg         =   self.agent['status']+", dealer's turn"
            status      =   2
            self.turn   =   'dealer'
        elif self.agent['status'] == 'On':
            msg         =   "Game is on, agent's turn"
            status      =   2
        # Check dealer's hand
        elif self.dealer['status'] == 'bust':
            msg         =   'agent wins, dealer busted'
            status      =   1
        elif self.agent['value'] < self.dealer['value']:
            msg         =   "agent loses, lower value"
            status      =   -1
        elif self.dealer['status'] == 'On':
            msg         =   "Game is on, dealer's turn"
            status      =   2
            self.turn   =   'dealer'
        else:
            # Compare hand value
            if self.agent['value'] > self.dealer['value']:
                msg         =   "agent wins, higher value"
                status      =   1
            else:
                msg         =   'Tied game'
                status      =   0
        if printStatus:
            self.status_print(msg, status)
        # Exit
        if statusOnly:
            return status
        if status < 2:
            # Episode over
            self.history.append(status)
            self.game_start()
        elif self.turn=='dealer' and self.turn!=oldTurn:
            # Dealer shows hidden card
            self.hand_do('hit')
        elif self.deck_empty:
            self.deck_new()

    def status_print(self, msg, status):
        # Common prints
        dlh =   []
        for ii, jj in zip(self.dealer['hand'], self.dealer['shown']):
            if jj:
                dlh.append(ii)
            else:
                dlh.append('Hidden')
        # Start print
        if msg=='New game':
            print('======GAME #' + str(len(self.history) + 1) + '======')
            print('Player'.ljust(12)+'Hand'.ljust(50)+'Value'.ljust(12)+'Status'.ljust(12)+'Game'.ljust(36))
            print('-'*110)
        else: # Print status
            # Print results
            print('Agent'.ljust(12)+', '.join(self.agent['hand']).ljust(50)+str(self.agent['value']).ljust(12)+self.agent['status'].ljust(12)+msg.ljust(36))
            print('Dealer'.ljust(12)+', '.join(dlh).ljust(50)+str(self.dealer['value']).ljust(12)+self.dealer['status'].ljust(12))
            # Game ending
            if status<2:
                print('======\n\n')
            else:
                print('\n')


# Demo
#game    =   blackJack('infinite')
#game    =   blackJack()
#game.hand_do('hit')
#game.hand_do('stick')

