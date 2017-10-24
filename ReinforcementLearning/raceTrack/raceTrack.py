""" This code solves the raceTrack problem in Sutton and Barto 5.6

    TO DO:
        -   make sure coordinate tuples are always stored as (y,x)
        -   2 racers cannot start on the same spot - to be fixed

"""

import numpy as np
import matplotlib.pyplot as plt
from random import choice
from ReinforcementLearning.raceTrack.racer import *

# Init interactive display
plt.ion()


class raceTrack():

    def __init__(self, trackType=1, display=True):
        # Generate track shape
        self.track_init(trackType)
        # Initialize racers
        self.racers     =   []
        # Initialize display
        self.figId      =   None
        self.display()

    def track_init(self, trackType):
        # First example
        if trackType==1:
            # Dimensions
            trHeight, trWidth   =   (32, 17)
            # Forbidden zones
            forbidden           =   [3,0]+[2,0]*2+[1,0]+[0,0]*2+[0,7]+[0,8]*7+[1,8]*8+[2,8]*7+[3,8]*3
            forbidden           =   [forbidden[(x*2):(x*2)+2] for x in range(int(len(forbidden)/2))]
            # Starting zone
            startZone           =   [list(range(4, 10)), [32]*6]
            # Finish zone
            finishZone          =   [[17]*6, list(range(1,7))]
        # Second example
        elif trackType==2:
            # Dimensions
            trHeight, trWidth   =   (30, 32)
            # Forbidden zones
            forbidden           =   [16, 0] + [13, 0] + [12,0] + [11, 0]*4 + [12,0] + [13,0] + [14, 2] + [14,5] + [14,6] + [14, 8] + [14, 9] + [13, 9] + [12, 9] + [11, 9] + [10, 9] + [9, 9] + [8, 9] + [7, 9] + [6, 9] + [5, 9] + [4, 9] + [3, 9] + [2, 9] + [1, 9] + [0, 9] + [0, 9] + [0, 9]
            forbidden           =   [forbidden[(x * 2):(x * 2) + 2] for x in range(int(len(forbidden) / 2))]
            # Starting zone
            startZone           =   [list(range(1, 24)), [30] * 23]
            # Finish zone
            finishZone          =   [[32] * 9, list(range(1, 10))]
        # Make track
        track_reward    =   [np.array([-5]*(x[0]+1)+[-1]*(trWidth-sum(x))+[-5]*(x[-1]+1)) for x in forbidden]
        track_reward    =   np.append( np.append( np.array([-5]*(trWidth+2), ndmin=2), track_reward, axis=0 ), np.array([-5]*(trWidth+2), ndmin=2), axis=0)
        # Make start
        for ii in range(np.shape(startZone)[-1]):
            track_reward[startZone[1][ii], startZone[0][ii]]    =   0
        # Make finish
        for ii in range(np.shape(finishZone)[-1]):
            track_reward[finishZone[1][ii], finishZone[0][ii]]  =   5
        # EOF
        self.track_start    =   startZone
        self.track_finish   =   finishZone
        self.track_dim      =   (trWidth, trHeight)
        self.track_reward   =   track_reward

    def track_pickStart(self):
        # Select at random
        idPick              =   choice( list(range(len(self.track_start[0]))) )
        return   [self.start[0][idPick], self.start[1][idPick]]

    def compute_displacement(self, position, velocity):
        # This function issues the new state and reward after displacement
        newPos      =   position + velocity
        newPos[0]   =   max( min(newPos[0], 0), self.track_dim[0] )
        newPos[1]   =   max( min(newPos[1], 0), self.track_dim[1] )
        # Compute reward
        reward      =   self.track_reward[newPos[0], newPos[1]]
        decrement   =   [0, 0]
        while reward==-5:
            # New decrement
            decrement   +=  [velocity[0]/sum(velocity), velocity[1]/sum(velocity)]
            # Walk back
            newPos2     =   newPos - [int(decrement[0]), int(decrement[1])]
            # New reward
            reward      =   self.track_reward[newPos2[0], newPos2[1]]
            if reward>-5:
                newPos  =   newPos2
        return reward, newPos

    def add_racer(self):
        # New racer
        self.racers.append(racer(self.track_pickStart(), [0,0], self.track_dim))

    def race_terminated(self, position):
        # Check if position is terminal
        terminated  =   False
        for iy,ix in zip(self.track_finish[0], self.track_finish[1]):
            terminated  =   terminated or [iy,ix]==position

    def race_run(self, nRaces):
        # Loop on number of races
        for iRc in range(nRaces):
            # Init the racers
            [x.__init__() for x in self.racers]
            race_on     =   [True]*len(self.racers)
            while any(race_on):
                # Compute displacement
                reward, newpos  =   [self.compute_displacement(self.racers[x].position_chain[-1], self.racers[x].velocity_chain[-1]) if y else [] for x,y in zip(range(len(self.racers)), race_on)]
                # Update racers
                [self.racers[w].car_update(x, y, self.race_terminated(y)) if z else [] for w,x,y,z in zip(range(len(self.racers)), newpos, reward, race_on)]

    def display(self):
        # Number of racers
        nRacers     =   len(self.racers)
        # Create figure
        if self.figId is None:
            self.figId  =   plt.figure()
            self.ax1    =   self.figId.add_subplot(100+(1+nRacers)*10+1)
        # ===========
        # TRACK MASKS
        # Mask1: raceTrack
        mask1   =   self.track_reward>-5
        # Mask2: starting zone
        mask2   =   np.transpose(np.zeros(np.add(self.track_dim,2)))
        for iX,iY in zip(self.track_start[1], self.track_start[0]):
            mask2[iX,iY]    =   .8
        # Mask3: Finish zone
        mask3   =   np.transpose(np.zeros(np.add(self.track_dim,2)))
        for iX, iY in zip(self.track_finish[1], self.track_finish[0]):
            mask3[iX, iY] = .4
        # Prep track
        IMtrack =   mask1+mask2+mask3
        IMtrack =   np.dstack([IMtrack]*3)
        # ===========
        # RACERS DOTS
        lsRGB   =   [[1,0,0], [1,.5,0], [1,1,0], [0.5, 1, 0], [0,1,0], [0,1,0.5], [0,1,1], [0,0.5,1], [0,0,1], [0.5,0,1], [1,0,1], [1,0,0.5]]
        for iRac in range(nRacers):
            x,y =   self.racers[iRac].position
            IMtrack[y,x,:]  =   lsRGB[iRac]
        # ===========
        # DRAW TRACK
        plt.imshow(IMtrack, interpolation='nearest', axes=self.ax1)
        # Minor ticks
        self.ax1.set_xticks(np.arange(-.5, self.track_dim[0]+2, 1), minor=True);
        self.ax1.set_yticks(np.arange(-.5, self.track_dim[1]+2, 1), minor=True);
        self.ax1.grid(which='minor', color='k', linestyle='-', linewidth=1)

# LAUNCHER
RT  =   raceTrack(trackType=1)