""" This class codes racers that can run on class: raceTrack.py

    TO DO:
        -   add the velocity dimension

    TO TEST:
        -   velocity never goes down to 0

"""

import numpy as np
from random import choice


class racer():

    def __init__(self, position, velocity, spaceDim, learnType='monte_carlo', learnRate=0.1, eGreedy=0, discount=0.8):
        # ==============
        # Learning agent
        # Agent type
        self.learnType      =   learnType
        self.learnRate      =   learnRate
        self.eGreedy        =   eGreedy
        self.discount       =   discount
        # Actions
        actY                =   [-1,0,1]*3
        actX                =   [-1]*3+[0]*3+[1]*3
        self.actions        =   [[actY[idx], actX[idx]] for idx in range(len(actX))]
        # Velocities
        velX                =   list(range(6))*6
        velY                =   [0]*6+[1]*6+[2]*6+[3]*6+[4]*6+[5]*6
        self.velocities     =   [[velY[idx], velX[idx]] for idx in range(len(velX))]
        # Space dimensions for learning
        self.policy         =   np.zeros(spaceDim+[len(self.velocities), len(self.actions)])
        self.action_value   =   np.zeros(spaceDim+[len(self.velocities), len(self.actions)])
        self.returns        =   np.zeros(spaceDim+[len(self.velocities), len(self.actions)])
        self.visits         =   np.zeros(spaceDim+[len(self.velocities), len(self.actions)])
        self.state_value    =   np.zeros(spaceDim)
        # Empty learning variables
        self.position_chain =   [position]
        self.action_chain   =   [self.velocities.index(velocity)]
        self.velocity_chain =   []
        self.cumul_reward   =   0
        self.car_control()

    def car_control(self):
        # ==============
        # Pick action: stochastic
        if len(self.action_chain)==0 and self.eGreedy==0:
            # First move must be exploratory
            x       =   list( range( len(self.actions) ) )
            x.pop(4)    # Ensure speed is never on cold start
            idX     =   choice( x )
        else:
            curPos  =   self.position_chain[-1]
            curVel  =   self.velocity_chain[-1]
            moveP   =   np.multiply(self.policy[curPos[0], curPos[1], curVel, :], np.random.random[1, len(self.actions)])
            crit1   =   [sum(x - curVel) > 0 for x in self.actions]
            crit2   =   [(x[0] - curVel[0]) >= 0 for x in self.actions]
            crit3   =   [(x[1] - curVel[1]) >= 0 for x in self.actions]
            moveP   =   [moveP[x] if y and z and w else 0 for x,y,z,w in zip(range(len(moveP)),crit1,crit2,crit3)]
            _,x     =   np.where( moveP==max(moveP) )
            idX     =   choice(range(len(x)))
        iAction     =   self.actions[x[idX]]
        # Set car acceleration/deceleration
        velocity    =   curVel + iAction
        velocity    =   [min(velocity[0],5), min(velocity[1],5)]
        self.action_chain.append(x[idX])
        self.velocity_chain.append(self.velocities.index(velocity))

    def car_set_policy(self, state, velocity):
        # ==============
        # Update action probabilities
        # Find the max
        values  =   self.action_value[state[0], state[1], velocity, :]
        nEl     =   len(values)
        iMax    =   np.argmax(values)
        # Greedy value
        self.policy[state[0], state[1], velocity, :]      =   self.eGreedy / nEl
        self.policy[state[0], state[1], velocity, iMax]   =   1 - self.eGreedy * (1 - 1 / nEl)

    def car_update(self, newPos, reward, terminated):
        # ==============
        # Time to learn
        self.cumul_reward += reward
        self.position_chain.append(newPos)
        self.car_control()
        # MONTE-CARLO LEARNING
        if self.learnType=='monteCarlo' and terminated:
            # Distribute reward to chain - First-visit
            firstVisit  =   {tuple(_key):True for _key in self.position_chain}
            for st, ac, ve in zip(self.position_chain[:-1], self.action_chain[:-1], self.velocity_chain[:-1]):
                if firstVisit[tuple(ac)]:
                    self.returns[st[0], st[1], ve, ac]      +=  self.cumul_reward
                    self.visits[st[0], st[1], ve, ac]       +=  1
                    # State value and policy
                    self.action_value[st[0], st[1], ve, ac] =   self.returns[st[0], st[1], ac] / self.visits[st[0], st[1], ac]
                    self.car_set_policy(st, ve)
                    # Not first visit anymore
                    firstVisit[tuple(ac)]   =   False
        # TEMPORAL-DIFFERENCE LEARNING
        elif self.learnType == 'TD0':
            Qold    =   self.action_value[self.position_chain[-2][0], self.position_chain[-2][1], self.velocity_chain[-2], self.action_chain[-2]]
            Qnew    =   self.action_value[self.position_chain[-1][0], self.position_chain[-1][1], self.velocity_chain[-1], self.action_chain[-1]]
            self.action_value[self.position_chain[-2][0], self.position_chain[-2][1], self.velocity_chain[-2], self.action_chain[-2]]  =   Qold + self.learnRate * (self.discount * Qnew - Qold)
            self.car_set_policy(self.position_chain[-2], self.velocity_chain[-2])
