""" This code solves the raceTrack problem in Sutton and Barto 5.6

    TO DO:
        -   No more racers than the number of starting blocks
        -   2 racers cannot start on the same spot - to be fixed
        -   Seen a case where car hits the finish line but the episode does not end

"""

import numpy as np
import pickle
import matplotlib
import cProfile
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utils.programming.ut_make_video import make_video
from random import choice
from time import sleep
from copy import deepcopy
from sys import stdout
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
        self.imageSeries=   []
        self.figId      =   None
        #self.display()
        #self.display(draw=True)
        #plt.pause(0.5)
        #plt.close(self.figId)
        #self.figId = None

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
            track_reward[finishZone[0][ii], finishZone[1][ii]]  =   5   #100
        # EOF
        self.track_start    =   startZone
        self.track_finish   =   finishZone
        self.track_dim      =   (trHeight+2, trWidth+2)
        self.track_reward   =   track_reward

    def track_pickStart(self):
        # Select at random
        idPick  =   choice( list(range(len(self.track_start[0]))) )
        return   [self.track_start[0][idPick], self.track_start[1][idPick]]

    def compute_displacement(self, position, velocity):
        # This function issues the new state and reward after displacement
        newPos      =   np.add(position, velocity)
        # Compute reward
        reward      =   self.track_reward[min( max(newPos[0], 0), self.track_dim[0]-1 ), min( max(newPos[1], 0), self.track_dim[1]-1 )]
        reward2     =   reward
        decrement   =   [0, 0]
        while reward2==-5 or any(newPos<0) or any(newPos>np.subtract(self.track_dim,1)):
            # New decrement
            decrement   =   np.add(decrement, [velocity[0]/sum(np.abs(velocity)), velocity[1]/sum(np.abs(velocity))])
            # Walk back
            newPos2     =   newPos - [int(decrement[0]), int(decrement[1])]
            # New reward
            reward2     =   self.track_reward[min( max(newPos2[0], 0), self.track_dim[0]-1 ), min( max(newPos2[1], 0), self.track_dim[1]-1 )]
            reward      +=  reward2
            if reward2>-5:
                newPos  =   newPos2
                velocity=   [0,0]           # Uncomment this line to have the car velocity set to 0 after hitting a wall
        return reward, newPos, [0,0]        # velocity

    def compute_FoV(self, racerInst, position):
        # Box seed
        padSize =   [int( (racerInst.viewY-1)/2 ), int( (racerInst.viewX-1)/2 )]
        bxSeed  =   np.subtract(position, padSize)
        bxBnd   =   np.add(position, padSize)
        # Make a empty FoV
        FoV     =   np.ones([racerInst.viewY, racerInst.viewX])*-5
        # Get the box
        box     =   self.track_reward[max(0,bxSeed[0]):min(self.track_dim[0],bxBnd[0]+1), max(0,bxSeed[1]):min(self.track_dim[1],bxBnd[1]+1)]
        bxShape =   box.shape
        # Put the box in FoV
        stX     =   abs( min(0,bxSeed[1]) )
        stY     =   abs( min(0,bxSeed[0]) )
        FoV[stY:(stY+bxShape[0]), stX:(stX+bxShape[1])]   =   box
        # Convert FoV to an integer
        FoV     =   FoV.flatten()
        FoVi    =   ['1' if x==-5 else '0' for x in FoV]
        FoVi    =   int(''.join(FoVi), 2)
        return FoVi

    def reset_racer(self, hRacer='new', Lambda=0, eGreedy=0.1, navMode='global'):
        # Pre-compute position
        position    =  self.track_pickStart()
        # New racer
        if hRacer=='new':
            self.racers.append(racer(list(self.track_dim), Lambda=Lambda, eGreedy=eGreedy, navMode=navMode))
            hRacer  =   self.racers[-1]
        initFoV     =   self.compute_FoV(hRacer, position)
        hRacer.car_set_start(position, [0, 0], FoV=initFoV)

    def race_terminated(self, position):
        # Check if position is terminal
        terminated      =   False
        for iy,ix in zip(self.track_finish[0], self.track_finish[1]):
            terminated  =   terminated or [iy,ix]==list(position)
        return terminated

    def race_run(self, nRaces, display=True, videoTape=None, pgbar=True):
        # Loop on number of races
        stepsBrace      =   np.zeros([len(self.racers), nRaces])
        rewBrace        =   np.zeros([len(self.racers), nRaces])
        locBrace        =   np.zeros([len(self.racers), nRaces])

        # Loop on number of races
        for iRc in range(nRaces):
            race_on     =   [True]*len(self.racers)

            # Count the steps
            while any(race_on):
                # Compute displacement
                rew_pos =   [self.compute_displacement(self.racers[x].position_chain[-1], self.racers[x].velocities[self.racers[x].velocity_chain[-1]]) if y else [] for x,y in zip(range(len(self.racers)), race_on)]
                # Update field of view
                fov     =   [self.compute_FoV(x, position[1]) for x,position in zip(self.racers, rew_pos)]
                # Update racers' learning
                [self.racers[w].car_update(list(x[1]), list(x[2]), z, x[0], self.race_terminated(x[1])) if y else [] for w,x,y,z in zip(range(len(self.racers)), rew_pos, race_on, fov)]
                # Update race status
                race_on  =  [not self.race_terminated(x[1]) for x in rew_pos]
                # Update display
                if display or not videoTape is None: self.display(draw=False)
            # Update progress bar
            if pgbar:
                stdout.write('\r')
                # the exact output you're looking for:
                msgR    =   ['Racer '+str(x+1)+': '+str(y.cumul_steps)+' steps' for x,y in zip(range(len(self.racers)), self.racers)]
                stdout.write("Running races: [%-40s] %d%%, %s" % ('=' * int(iRc / nRaces * 40), 100 * iRc / nRaces, ', '.join(msgR)))
                stdout.flush()
            # Pick new starting positions
            stepsBrace[:,iRc]   =   [x.cumul_steps for x in self.racers]
            rewBrace[:,iRc]     =   [x.cumul_reward for x in self.racers]
            locBrace[:,iRc]     =   [np.mean(x.cumul_locWeight) for x in self.racers]
            [self.reset_racer(hRacer=x) for x in self.racers]
        # Close display
        if display or not videoTape is None:
            self.display(draw=True, videoTape=videoTape)
            plt.close(self.figId)
            self.figId          =   None
        return stepsBrace, rewBrace, locBrace

    def race_log(self, steps, iterations, pgbOn=True):
        # Prepare containers
        nRaces  =   [1] + [steps]*iterations
        Qlog    =   {'nRaces':[0], 'nSteps':[], 'reward':[], 'locWgt':[], 'currentPolicies':[]}
        # Loop on iterations
        count   =   0
        for nIt in nRaces:
            STP, REW, LOC    =   self.race_run(nIt, display=False, pgbar=pgbOn)
            Qlog['nRaces'].append( nIt + Qlog['nRaces'][-1] )
            Qlog['nSteps'].append( np.mean(STP, axis=1) )
            Qlog['reward'].append( np.mean(REW, axis=1) )
            Qlog['locWgt'].append( np.mean(LOC, axis=1) )
            Qlog['currentPolicies'].append([deepcopy(x.policy) for x in self.racers])

            # Print status
            count += 1
            stdout.write('\r')
            # the exact output you're looking for:
            stdout.write("Running iteration: [%-40s] %d%%, completed in %i steps on average" % ('=' * int(count / len(nRaces) * 40), 100 * count / len(nRaces), int(np.mean(STP, axis=1))))
            stdout.flush()
        return Qlog

    def display(self, draw=False, videoTape=None):

        # Number of racers
        nRacers = len(self.racers)

        def update_matrix(num, matrix, hndl):
            hndl.set_data( matrix[num] )
            return hndl,

        if not draw:
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
            # NO DISPLAY MODE
            self.imageSeries.append(IMtrack)
            return

        # ===========
        # CREATE FIGURE
        self.figId  =   plt.figure()
        self.ax1    =   self.figId.add_subplot(100 + (1 + nRacers) * 10 + 1)
        # DRAW TRACK
        showId      =   plt.imshow(self.imageSeries[0], interpolation='nearest', axes=self.ax1)

        # Adjust the figure size
        axPos       =   self.ax1.get_position()
        self.figId.set_size_inches( self.figId.get_size_inches() * [axPos.width, axPos.height] )
        self.ax1.set_position([0.2, 0.1, 0.7, 0.8])

        # Minor ticks
        self.ax1.set_xticks(np.arange(-.5, self.track_dim[1], 1), minor=True);
        self.ax1.set_yticks(np.arange(-.5, self.track_dim[0], 1), minor=True);
        self.ax1.grid(which='minor', color='k', linestyle='-', linewidth=1)

        if not videoTape is None:
            # Initiate writer
            Writer      =   animation.writers['ffmpeg']
            writer      =   Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
            line_ani    =   animation.FuncAnimation(self.figId, update_matrix, len(self.imageSeries), fargs=(self.imageSeries, showId),blit=False)
            line_ani.save(videoTape, writer=writer)
        else:
            while len(self.imageSeries)>0:
                showId.set_data(self.imageSeries.pop(0))
                plt.show()
                plt.draw()
                plt.pause(0.1)




# ========
# LAUNCHER
RT      =   raceTrack(trackType=1)
parLamb =   0.9
pareGr  =   0.1

"""
RT.reset_racer(hRacer='new', eGreedy=pareGr, Lambda=parLamb, navMode='global')
RT.race_run(1, display=False);

RT.reset_racer(hRacer='new', eGreedy=pareGr, Lambda=parLamb)
for ii in range(100):
    RT.racers[0].car_set_start([20,9], choice(RT.racers[0].velocities), RT.compute_FoV(RT.racers[0], [20,9]))
    RT.race_train(1, display=False, pgbar=True)
"""

#RT.race_run(1, display=True)
#RT.race_run(100, display=False, pgbar=False)

"""

# ==================
# COMPARE LAMBDAs
# ==================
# Iterate over lambda parameters for a single navigation mode
FF      =   plt.figure();
lVal    =   [0, .2, .4, .6, .8, .9, 1]
lCol    =   ['r', 'm', 'y', 'c', 'g', 'b', 'k']
Qlog_0  =   []
ax1     =   FF.add_subplot(121); ax1.title.set_text('Cumulative reward');   ax1.set_xlabel('Nb of races')
ax2     =   FF.add_subplot(122); ax2.title.set_text('Avg number of steps'); ax2.set_xlabel('Nb of races')
for iL, iC in zip(lVal, lCol):
    RT.reset_racer(hRacer='new', eGreedy=pareGr, Lambda=iL)
    Qlog_0.append( RT.race_log(50, 40, pgbOn=False) )
    ax1.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['reward'], iC, label='lambda: '+str(iL))
    ax2.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['nSteps'], iC, label='lambda: '+str(iL))
    plt.pause(0.5)
    RT.racers.pop()
ax1.legend()
ax1.set_xlim([1,2000]); ax1.set_ylim([-500,5])
ax2.set_xlim([1,2000]); ax2.set_ylim([25,500])


# ==================
# COMPARE NAVI MODES
# ==================
# Iterate over navigation modes for a single lambda
FF  =   plt.figure()
ax1 =   FF.add_subplot(131); ax1.title.set_text('Cumulative reward');   ax1.set_xlabel('Nb of races')
ax2 =   FF.add_subplot(132); ax2.title.set_text('Avg number of steps'); ax2.set_xlabel('Nb of races')
ax3 =   FF.add_subplot(133); ax3.title.set_text('Weight of local info'); ax3.set_xlabel('Nb of races')
navM    =   ['global', 'sum', 'entropyWsum', 'maxAbs', 'local']
lCol    =   ['r', 'g', 'c', 'b', 'k']
Qlog_0  =   []
for iL, iC in zip(navM, lCol):
    RT.reset_racer(hRacer='new', eGreedy=pareGr, Lambda=parLamb, navMode=iL)
    print('Navigation mode: '+iL)
    Qlog_0.append( RT.race_log(20, 100, pgbOn=False) )
    ax1.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['reward'], iC, label='nav. mode: '+iL)
    ax2.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['nSteps'], iC, label='nav. mode: ' + iL)
    ax3.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['locWgt'], iC, label='nav. mode: ' + iL)
    plt.pause(0.5)
    RT.racers.pop()
ax1.legend()
ax1.set_xlim([1,2000]); ax1.set_ylim([-500,0])
ax2.set_xlim([1,2000]); ax2.set_ylim([25,300])
ax3.set_xlim([1,2000]); ax3.set_ylim([.5, .8])

#SAVE
logVar      =   {'log':Qlog_0, 'figure':RT.figId}
wrkRep      =   '/home/younesz/Documents/Simulations/raceTrack/Type1/'
filename    =   '1Racer_compareNavModes_lambda'+str(parLamb).replace('.', '_')+'_eGreedy'+str(pareGr).replace('.', '_')+'_10Kraces.p'
with open(wrkRep+filename, 'wb') as f:
    pickle.dump(logVar, f)
    
    

# ==================
# COMPARE SIMULTANEOUS vs SEQUENTIAL LEARNING
# ==================
FF  =   plt.figure()
ax1 =   FF.add_subplot(121); ax1.title.set_text('Cumulative reward');   ax1.set_xlabel('Nb of races')
ax2 =   FF.add_subplot(122); ax2.title.set_text('Avg number of steps'); ax2.set_xlabel('Nb of races')
Qlog_0 =    []
# ---- First do sequential
# LOCAL training
RT.reset_racer(hRacer='new', eGreedy=pareGr, Lambda=parLamb, navMode='local')
print('Sequential learning')
Qlog_0.append( RT.race_log(25, 100, pgbOn=False) )
#ax1.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['reward'], 'r--', label='localOnly')
#ax2.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['nSteps'], 'r--', label='localOnly')
# GLOBAL training
RT.racers[0].navMode    =   'global'
Qlog_0.append( RT.race_log(25, 100, pgbOn=False) )
ax1.plot( np.add(Qlog_0[-1]['nRaces'][1:],2500), Qlog_0[-1]['reward'], 'r', label='globalOnly')
ax2.plot( np.add(Qlog_0[-1]['nRaces'][1:],2500), Qlog_0[-1]['nSteps'], 'r', label='globalOnly')
# ---- Next do simultaneous
print('Simultaneous learning: sum')
RT.racers.pop()
RT.reset_racer(hRacer='new', eGreedy=pareGr, Lambda=parLamb, navMode='sum')    
Qlog_0.append( RT.race_log(25, 200, pgbOn=False) )
ax1.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['reward'], 'b', label='l+g: sum')
ax2.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['nSteps'], 'b', label='l+g: sum')    
print('Simultaneous learning: maxAbs')
RT.racers.pop()
RT.reset_racer(hRacer='new', eGreedy=pareGr, Lambda=parLamb, navMode='maxAbs')    
Qlog_0.append( RT.race_log(25, 200, pgbOn=False) )
ax1.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['reward'], 'k', label='l+g: sum')
ax2.plot(Qlog_0[-1]['nRaces'][1:], Qlog_0[-1]['nSteps'], 'k', label='l+g: sum')    
    
ax1.legend()
ax1.set_xlim([1,50000]); ax1.set_ylim([-5000,0])
ax2.set_xlim([1,50000]); ax2.set_ylim([0,300])
ax3.set_xlim([1,50000]); ax3.set_ylim([.5, .8])

#SAVE
logVar      =   {'log':Qlog_0, 'figure':RT.figId}
wrkRep      =   '/home/younesz/Documents/Simulations/raceTrack/Type1/'
filename    =   '1Racer_compareNavModes_sequentialVSsimultaneous_lambda'+str(parLamb).replace('.', '_')+'_eGreedy'+str(pareGr).replace('.', '_')+'_10Kraces.p'
with open(wrkRep+filename, 'wb') as f:
    pickle.dump(logVar, f)    
    
    
# ==================
# DUMP, essentially
# ==================    
            
#SAVE
logVar      =   {'log':Qlog_0, 'figure':RT.figId}
wrkRep      =   '/home/younesz/Documents/Simulations/raceTrack/Type1/'
filename    =   '1Racer2VisionsSumLog_TD'+str(parLamb).replace('.', '_')+'_eGreedy'+str(pareGr).replace('.', '_')+'_10Kraces.p'
with open(wrkRep+filename, 'wb') as f:
    pickle.dump(logVar, f)


import pickle
with open('/home/younesz/Documents/Simulations/raceTrack/Type1/1RacerLog_TD010K_races', 'rb') as f:
    Qlog = pickle.load(f)
RT.racers[0].learnType = 'noLearning'


# After 1 race
RT.racers[0].policy = Qlog['racerMirror'][1][0].policy
RT.race_run(1, display=True, videoTape='/home/younesz/Documents/Simulations/raceTrack/Type1/Run_1race_iter1.mp4')
# After 5K races
RT.racers[0].policy = Qlog['racerMirror'][100][0].policy
RT.race_run(1, display=True, videoTape='/home/younesz/Documents/Simulations/raceTrack/Type1/Run_5Kraces_iter1.mp4')
# After 10K races
RT.racers[0].policy = Qlog['racerMirror'][-1][0].policy
RT.race_run(1, display=False, videoTape='/home/younesz/Documents/Simulations/raceTrack/Type1/Run_10Kraces_iter1.mp4')
"""
