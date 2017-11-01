""" This class codes racers that can run on class: raceTrack.py

    TO DO:
        -   restrict velocity to positive values

    TO TEST:
        -   velocity never goes down to 0

"""

import numpy as np
from random import choice


class racer():

    def __init__(self, position, velocity, spaceDim, Lambda=0, learnRate=0.15, eGreedy=0.1, discount=0.8):
        # ==============
        # Learning agent
        # Agent type
        self.Lambda         =   Lambda
        self.learnRate      =   learnRate
        self.eGreedy        =   eGreedy
        self.discount       =   discount
        # Actions
        actY                =   [-1,0,1]*3
        actX                =   [-1]*3+[0]*3+[1]*3
        self.actions        =   [[actY[idx], actX[idx]] for idx in range(len(actX))]
        # Velocities
        velX                =   list(range(-5,6))*11
        velY                =   [-5]*11+[-4]*11+[-3]*11+[-2]*11+[-1]*11+[0]*11+[1]*11+[2]*11+[3]*11+[4]*11+[5]*11
        self.velocities     =   [[velY[idx], velX[idx]] for idx in range(len(velX))]
        # Field of view
        self.viewX          =   5
        self.viewY          =   5
        # Space dimensions for learning
        self.policy         =   np.zeros(spaceDim+[len(self.velocities), len(self.actions)])
        self.global_value   =   np.zeros(spaceDim+[len(self.velocities), len(self.actions)])
        self.local_value    =   {}
        # Initialize starting position
        self.car_set_start(position, velocity)
        # Print message
        print('Initialized 1 racer')

    def car_set_start(self, position, velocity):
        # Empty learning variables
        self.position_chain =   [position]
        self.action_chain   =   []
        self.velocity_chain =   [self.velocities.index(velocity)]
        self.global_trace   =   np.zeros( np.shape(self.global_value) )
        self.local_trace    =   {}
        self.cumul_reward   =   0
        self.car_control()

    def car_control(self):
        # ==============
        # Pick action: stochastic
        curVel = self.velocity_chain[-1]
        if len(self.action_chain)==0 and self.eGreedy==0:
            # First move must be exploratory
            x       =   list( range( len(self.actions) ) )
            x.pop(4)    # Ensure speed is never on cold start
            idX     =   choice( x )
        else:
            curPos  =   self.position_chain[-1]
            moveP   =   np.multiply(self.policy[curPos[0], curPos[1], curVel, :], np.random.random([1, len(self.actions)]))[0]
            crit1   =   [abs(sum(np.add(x, self.velocities[curVel]))) > 0 for x in self.actions]
            #crit2   =   [(x[0] + self.velocities[curVel][0]) >= 0 for x in self.actions]
            #crit3   =   [(x[1] + self.velocities[curVel][1]) >= 0 for x in self.actions]
            #moveP   =   [x if y and z and w else -1 for x,y,z,w in zip(list(moveP),crit1,crit2,crit3)]
            moveP   =   [x if y else -1 for x, y in zip(list(moveP), crit1)]
            x       =   np.where( np.array(moveP)==np.array(max(moveP)) )[0]
            idX     =   choice(x)
        iAction     =   self.actions[idX]
        # Set car acceleration/deceleration
        velocity    =   np.add(self.velocities[curVel], iAction)
        velocity    =   [max(min(velocity[0],5),-5), max(min(velocity[1],5),-5)]
        self.action_chain.append(idX)
        self.velocity_chain.append(self.velocities.index(velocity))

    def car_set_policy(self, state, FoV, velocity):
        # ==============
        # Update action probabilities
        # Find the max
        valuesG =   self.global_value[state[0], state[1], velocity, :]
        valuesL =   self.local_value[FoV][velocity, :]
        values  =   valuesG * valuesL
        nEl     =   len(values)
        # Pick a max randomly
        x       =   np.where( values==max(values) )
        iMax    =   choice(x[0])
        # Greedy value
        self.policy[state[0], state[1], velocity, :]      =   self.eGreedy / nEl
        self.policy[state[0], state[1], velocity, iMax]   =   1 - self.eGreedy * (1 - 1 / nEl)

    def car_update(self, newPos, newVelo, newFoV, reward, terminated):
        # ==============
        # Time to learn
        self.cumul_reward += reward
        self.position_chain.append(newPos)
        self.velocity_chain[-1]     =   self.velocities.index(newVelo)     # Set to 0,0 in case car hits a wall
        self.car_control()
        # SARSA lambda - global
        Qold    =   self.global_value[self.position_chain[-2][0], self.position_chain[-2][1], self.velocity_chain[-3], self.action_chain[-2]]
        Qnew    =   self.global_value[self.position_chain[-1][0], self.position_chain[-1][1], self.velocity_chain[-2], self.action_chain[-1]]
        incr    =   reward + self.discount * Qnew - Qold
        self.global_trace[self.position_chain[-2][0], self.position_chain[-2][1], self.velocity_chain[-3], self.action_chain[-2]]    +=  1
        self.global_value   +=  self.learnRate * incr * self.global_trace
        self.global_trace   *=  self.discount * self.Lambda
        # SARSA lambda - local
        FoVi    =   int(newFoV, 2)
        if not FoVi in self.local_value.keys():
            self.local_value[FoVi]  =   np.zeros( len(self.velocities), len(self.actions) )
            self.local_trace[FoVi]  =   np.zeros( len(self.velocities), len(self.actions) )
        Qold    =   self.local_value[FoVi][self.velocity_chain[-3], self.action_chain[-2]]
        Qnew    =   self.local_value[FoVi][self.velocity_chain[-2], self.action_chain[-1]]
        incr    =   reward + self.discount * Qnew - Qold
        self.local_trace[FoVi][self.velocity_chain[-3], self.action_chain[-2]] += 1
        for iK in self.local_value['keys']:
            self.local_value[iK]    +=  self.learnRate * incr * self.local_trace[iK]
            self.local_trace[iK]    *=  self.discount * self.Lambda
        # Update racers' policies
        self.car_set_policy(self.position_chain[-2], newFoV, self.velocity_chain[-3])

        """
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
            Qold    =   self.action_value[self.position_chain[-2][0], self.position_chain[-2][1], self.velocity_chain[-3], self.action_chain[-2]]
            Qnew    =   self.action_value[self.position_chain[-1][0], self.position_chain[-1][1], self.velocity_chain[-2], self.action_chain[-1]]
            self.action_value[self.position_chain[-2][0], self.position_chain[-2][1], self.velocity_chain[-3], self.action_chain[-2]]  =   Qold + self.learnRate * (reward + self.discount * Qnew - Qold)
            self.car_set_policy(self.position_chain[-2], self.velocity_chain[-3])
        """








