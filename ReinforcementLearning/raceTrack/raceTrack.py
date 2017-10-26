""" This code solves the raceTrack problem in Sutton and Barto 5.6

    TO DO:
        -   2 racers cannot start on the same spot - to be fixed

"""

import numpy as np
import matplotlib.pyplot as plt
from random import choice
from time import sleep
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
            startZone           =   [[32]*6, list(range(4, 10))]
            # Finish zone
            finishZone          =   [list(range(1,7)), [17]*6]
        # Second example
        elif trackType==2:
            # Dimensions
            trHeight, trWidth   =   (30, 32)
            # Forbidden zones
            forbidden           =   [16, 0] + [13, 0] + [12,0] + [11, 0]*4 + [12,0] + [13,0] + [14, 2] + [14,5] + [14,6] + [14, 8] + [14, 9] + [13, 9] + [12, 9] + [11, 9] + [10, 9] + [9, 9] + [8, 9] + [7, 9] + [6, 9] + [5, 9] + [4, 9] + [3, 9] + [2, 9] + [1, 9] + [0, 9] + [0, 9] + [0, 9]
            forbidden           =   [forbidden[(x * 2):(x * 2) + 2] for x in range(int(len(forbidden) / 2))]
            # Starting zone
            startZone           =   [[30] * 23, list(range(1, 24))]
            # Finish zone
            finishZone          =   [list(range(1, 10)), [32] * 9]
        # Make track
        track_reward    =   [np.array([-5]*(x[0]+1)+[-1]*(trWidth-sum(x))+[-5]*(x[-1]+1)) for x in forbidden]
        track_reward    =   np.append( np.append( np.array([-5]*(trWidth+2), ndmin=2), track_reward, axis=0 ), np.array([-5]*(trWidth+2), ndmin=2), axis=0)
        # Make start
        for ii in range(np.shape(startZone)[-1]):
            track_reward[startZone[0][ii], startZone[1][ii]]    =   -1  #This avoids that car switching between starting position at race onset
        # Make finish
        for ii in range(np.shape(finishZone)[-1]):
            track_reward[finishZone[0][ii], finishZone[1][ii]]  =   5
        # EOF
        self.track_start    =   startZone
        self.track_finish   =   finishZone
        self.track_dim      =   (trHeight+2, trWidth+2)
        self.track_reward   =   track_reward

    def track_pickStart(self):
        # Select at random
        idPick              =   choice( list(range(len(self.track_start[0]))) )
        return   [self.track_start[0][idPick], self.track_start[1][idPick]]

    def compute_displacement(self, position, velocity):
        # This function issues the new state and reward after displacement
        newPos      =   np.add(position, velocity)
        newPos[0]   =   min( max(newPos[0], 0), self.track_dim[0]-1 )
        newPos[1]   =   min( max(newPos[1], 0), self.track_dim[1]-1 )
        # Compute reward
        reward      =   self.track_reward[newPos[0], newPos[1]]
        reward2     =   reward
        decrement   =   [0, 0]
        while reward2==-5 or any(newPos<0) or any(newPos>np.subtract(self.track_dim,1)):
            # New decrement
            decrement   =   np.add(decrement, [velocity[0]/sum(np.abs(velocity)), velocity[1]/sum(np.abs(velocity))])
            # Walk back
            newPos2     =   newPos - [int(decrement[0]), int(decrement[1])]
            # New reward
            reward2     =   self.track_reward[newPos2[0], newPos2[1]]
            reward      +=  reward2
            if reward2>-5:
                newPos  =   newPos2
        return reward, newPos

    def add_racer(self):
        # New racer
        self.racers.append(racer(self.track_pickStart(), [0,0], list(self.track_dim), learnType='TD0'))

    def race_terminated(self, position):
        # Check if position is terminal
        terminated      =   False
        for iy,ix in zip(self.track_finish[0], self.track_finish[1]):
            terminated  =   terminated or [iy,ix]==list(position)
        return terminated

    def race_run(self, nRaces, display=True):
        # Loop on number of races
        for iRc in range(nRaces):
            race_on     =   [True]*len(self.racers)
            if display: self.display()
            while any(race_on):
                # Compute displacement
                rew_pos  =  [self.compute_displacement(self.racers[x].position_chain[-1], self.racers[x].velocities[self.racers[x].velocity_chain[-1]]) if y else [] for x,y in zip(range(len(self.racers)), race_on)]
                # Update racers
                [self.racers[w].car_update(list(x[1]), x[0], self.race_terminated(x[1])) if y else [] for w,x,y in zip(range(len(self.racers)), rew_pos, race_on)]
                # Update race status
                race_on  =  [not self.race_terminated(x[1]) for x in rew_pos]
                # Update display
                if display: self.display()

    def display(self):
        # Number of racers
        nRacers     =   len(self.racers)
        # ===========
        # TRACK MASKS
        # Mask1: raceTrack
        mask1   =   self.track_reward>-5
        # Mask2: starting zone
        mask2   =   np.zeros(self.track_dim)
        for iX,iY in zip(self.track_start[0], self.track_start[1]):
            mask2[iX,iY]    =   .8
        # Mask3: Finish zone
        mask3   =   np.zeros(self.track_dim)
        for iX, iY in zip(self.track_finish[0], self.track_finish[1]):
            mask3[iX, iY] = .4
        # Prep track
        IMtrack =   mask1+mask2+mask3
        IMtrack =   np.dstack([IMtrack]*3)
        # ===========
        # RACERS DOTS
        lsRGB   =   [[1,0,0], [1,.5,0], [1,1,0], [0.5, 1, 0], [0,1,0], [0,1,0.5], [0,1,1], [0,0.5,1], [0,0,1], [0.5,0,1], [1,0,1], [1,0,0.5]]
        for iRac in range(nRacers):
            y,x =   self.racers[iRac].position_chain[-1]
            IMtrack[y,x,:]  =   lsRGB[iRac]
        # ===========
        # CREATE FIGURE
        if self.figId is None:
            self.figId  =   plt.figure()
            self.ax1    =   self.figId.add_subplot(100 + (1 + nRacers) * 10 + 1)
            # DRAW TRACK
            self.showId =   plt.imshow(IMtrack, interpolation='nearest', axes=self.ax1)
        else:
            self.showId.set_data(IMtrack)
        # Minor ticks
        plt.show()
        plt.draw()
        self.ax1.set_xticks(np.arange(-.5, self.track_dim[1], 1), minor=True);
        self.ax1.set_yticks(np.arange(-.5, self.track_dim[0], 1), minor=True);
        self.ax1.grid(which='minor', color='k', linestyle='-', linewidth=1)
        plt.pause(0.2)


# LAUNCHER
RT  =   raceTrack(trackType=1)
RT.add_racer()
RT.race_run(1)
RT.race_run(100, display=False)