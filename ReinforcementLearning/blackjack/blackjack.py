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
"""

#==== IMPORTS
import numpy as np
from random import shuffle


class blackJack:

    def __init__(self, gameType='infinite'):
        # Data location
        self.gameType   =   gameType
        # Initiate game deck
        self.deck_new()

    def deck_new(self):
        # Make a new card deck
        colors      =   ['Heart', 'Diamond', 'Club', 'Spade']
        numbers     =   [str(x + 1) for x in range(10)] + ['jack', 'queen', 'king']
        self.deck   =   list( np.concatenate([[x+'_'+y for x in numbers] for y in colors][:]) )
        shuffle(self.deck)

    def game_start(self):
        # Check if truly new game
        if len(self.deck)!=52:
            print('\tAlready in the middle of a game, must exit first')
            return
        # Dealer gives two cards to agent
        self.agent_cards    =   self.deck.pop(0) + self.deck.pop(0)
        self.agent_shown    =   [True] * 2
        # Dealer gives two cards to himself
        self.dealer_cards   =   self.deck.pop(0) + self.deck.pop(0)
        self.dealer_shown   =   [True, False]

    def game_status(self):
        # Check for

    def dealer_clear_table(self):
        # Put cards back in the deck
        self.deck   +=  self.agent_cards
        self.deck   +=  self.dealer_cards
        # Remove them from hands
        self.agent_cards    =   []
        self.dealer_cards   =   []

    def agent_hit(self):
        # Take a card from the deck and put into agent's head
        self.agent_cards    +=  self.deck.pop(0)
        # Evaluate game status