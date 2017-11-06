""" This class codes racers that can run on class: raceTrack.py

    TO DO:
        -   correct the logging function to include multiple realizations at low nb of races

    TO TEST:
        -   tracking contribution of local info
        -   new combination of Q-values: entropy-weighted sum

"""

import numpy as np
from random import choice
from scipy.stats import entropy
from Utils.programming import ut_remove_value


class racer():

    def __init__(self, spaceDim, Lambda=0, learnRate=0.15, eGreedy=0.1, discount=0.8, navMode='global'):
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
        self.navMode        =   navMode
        # Print message
        print('Initialized 1 racer')

    def car_set_start(self, position, velocity, FoV=None):
        # Empty learning variables
        self.position_chain =   [position]
        self.action_chain   =   []
        self.FoV_chain      =   []
        if FoV is not None:
            self.FoV_chain  =   [FoV]
        if FoV not in self.local_value.keys():
            self.local_value[FoV]   =   np.zeros([len(self.velocities), len(self.actions)])
        self.velocity_chain =   [self.velocities.index(velocity)]
        self.global_trace   =   np.zeros( np.shape(self.global_value) )
        self.cumul_reward   =   0
        self.cumul_steps    =   0
        self.cumul_locWeight=   []
        # Reset local trace
        self.local_trace    =   {}
        for iTr in self.local_value.keys():
            self.local_trace[iTr]   =   np.zeros([len(self.velocities), len(self.actions)])
        self.car_control()

    def car_init_FoV(self, FoV):
        self.FoV_chain.append(FoV)
        # Initialize value within field of view
        self.local_trace[FoV]       =   {_key:np.zeros([len(self.velocities), len(self.actions)]) for _key in self.local_trace.keys()}
        if not FoV in self.local_value.keys():
            self.local_value[FoV]   =   np.zeros([len(self.velocities), len(self.actions)])

    def car_control(self):
        # ==============
        # Pick action: stochastic
        curVel      =   self.velocity_chain[-1]
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

        # ******** COMBINATION OF LOCAL AND GLOBAL INFORMATION
        if self.navMode=='global':
            # Only global information
            values  =   valuesG
            self.cumul_locWeight.append(0)
        elif self.navMode=='local':
            values  =   valuesL
            self.cumul_locWeight.append(1)
        elif self.navMode == 'sum':
            # Sum of both
            values  =   valuesG + valuesL
        elif self.navMode == 'entropyWsum':
            # entropy-weighted sum of both
            entrL   =   entropy( ut_remove_value.main(valuesL, '!=0') ) + 1
            entrG   =   entropy( ut_remove_value.main(valuesG, '!=0') ) + 1
            values  =   valuesG/entrG + valuesL/entrL
        elif self.navMode == 'maxAbs':
            # Max of absolute value
            mxV     =   np.maximum(np.abs(valuesG), np.abs(valuesL))
            snV     =   np.sign
            values  =   [[x,y][np.abs([x,y]).argmax()] for x,y in zip(valuesG, valuesL)]

        nEl     =   len(values)
        # Pick a max randomly
        x       =   np.where( values==max(values) )
        iMax    =   choice(x[0])
        # Greedy value
        self.policy[state[0], state[1], velocity, :]      =   self.eGreedy / nEl
        self.policy[state[0], state[1], velocity, iMax]   =   1 - self.eGreedy * (1 - 1 / nEl)
        # Update contributions of global vs local
        if self.navMode == 'sum':
            if (abs(valuesG[iMax]) + abs(valuesL[iMax])) == 0:
                self.cumul_locWeight.append(0.5)
            else:
                self.cumul_locWeight.append( abs(valuesL[iMax]) / (abs(valuesG[iMax]) + abs(valuesL[iMax])) )
        elif self.navMode == 'entropyWsum':
            if (abs(valuesG[iMax])/entrG + abs(valuesL[iMax])/entrL)==0:
                self.cumul_locWeight.append(0.5)
            else:
                self.cumul_locWeight.append( (abs(valuesL[iMax])/entrL) / (abs(valuesG[iMax])/entrG + abs(valuesL[iMax])/entrL) )
        elif self.navMode == 'maxAbs':
            self.cumul_locWeight.append( abs(valuesL[iMax])==max(valuesL[iMax],valuesG[iMax])  - abs(valuesG[iMax])==max(valuesL[iMax],valuesG[iMax]))

    def car_update(self, newPos, newVelo, newFoV, reward, terminated):
        # ==============
        self.cumul_steps    +=  1
        self.cumul_reward   +=  reward
        self.position_chain.append(newPos)
        # Append field of view
        self.FoV_chain.append(newFoV)
        if not self.FoV_chain[-1] in self.local_value.keys():
            self.local_value[self.FoV_chain[-1]] = np.zeros([len(self.velocities), len(self.actions)])
            self.local_trace[self.FoV_chain[-1]] = np.zeros([len(self.velocities), len(self.actions)])
        #self.velocity_chain[-1]     =   self.velocities.index(newVelo)     # Set to 0,0 in case car hits a wall
        self.car_control()
        # Global Learning
        if self.navMode != 'local':
            # SARSA lambda - global
            Qold    =   self.global_value[self.position_chain[-2][0], self.position_chain[-2][1], self.velocity_chain[-3], self.action_chain[-2]]
            Qnew    =   self.global_value[self.position_chain[-1][0], self.position_chain[-1][1], self.velocity_chain[-2], self.action_chain[-1]]
            incr    =   reward + self.discount * Qnew - Qold
            self.global_trace[self.position_chain[-2][0], self.position_chain[-2][1], self.velocity_chain[-3], self.action_chain[-2]]    +=  1
            self.global_value   +=  self.learnRate * incr * self.global_trace
            self.global_trace   *=  self.discount * self.Lambda
        # Local learning
        if self.navMode != 'global':
            # SARSA lambda - local
            Qold    =   self.local_value[self.FoV_chain[-2]][self.velocity_chain[-3], self.action_chain[-2]]
            Qnew    =   self.local_value[self.FoV_chain[-1]][self.velocity_chain[-2], self.action_chain[-1]]
            incr    =   reward + self.discount * Qnew - Qold
            self.local_trace[self.FoV_chain[-2]][self.velocity_chain[-3], self.action_chain[-2]] = 1
            for iK in self.local_value.keys():
                self.local_value[iK]    +=  self.learnRate * incr * self.local_trace[iK]
                self.local_trace[iK]    *=  self.discount * self.Lambda
        # Update racers' policies
        self.car_set_policy(self.position_chain[-2], self.FoV_chain[-2], self.velocity_chain[-3])

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








