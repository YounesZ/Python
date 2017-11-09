# This class implements various maze environments

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import choice


class maze():

    # INIT FUNCTIONS
    # ==============
    def __init__(self, type='template1', params=(0,0,1)):
        # Make the maze
        self.maze_make(type, params)
        # Display variables
        self.figId          =   None
        self.imagePanels    =   []
        self.arrowsP        =   [[[] for x in range(self.maze_dims[1])] for y in range(self.maze_dims[0])]
        self.arrowsV        =   [[[] for x in range(self.maze_dims[1])] for y in range(self.maze_dims[0])]
        # Agents variables
        self.agents         =   []

    def maze_make(self, type, params):
        # Common variables
        onMaze_rew  =   params[0]       # Penalty for transitions
        offMaze_rew =   params[1]       # Penalty for trying to quit maze
        finish_rew  =   params[2]       # Reward foe finishing the maze

        # Template 1:   Sutton and Barto, p.191
        if type=='template1':
            h, w    =   6, 9
            start   =   [[2,0]]
            finish  =   [[0,8]]
            obstacle=   [(1,2,2,0), (4,5,0,0), (0,7,2,0)]

        # --- Make the maze
        # Regular transitions
        allowed     =   np.ones([h, w])
        reward      =   np.zeros([h,w]) + onMaze_rew
        # Finish line
        for ifin in finish:
            reward[ifin[0],ifin[1]]     =   finish_rew
        # Obstacles
        for iobs in obstacle:
            reward[iobs[0]:iobs[0]+iobs[2]+1, iobs[1]:iobs[1]+iobs[3]+1]            =   offMaze_rew
            allowed[iobs[0]:iobs[0] + iobs[2] + 1, iobs[1]:iobs[1] + iobs[3] + 1]   =   0
        # Starting grid
        for ist in start:
            allowed[ist[0],ist[1]]  =   -1

        # --- Store maze
        self.maze_allowed   =   allowed
        self.maze_dims      =   [h,w]
        self.maze_start     =   start
        self.maze_finish    =   finish
        self.maze_reward    =   reward
        self.move_reward    =   params

    # STATE FUNCTIONS
    # ===============
    def compute_displacement(self, position, move):
        # This function issues the new state and reward after displacement
        newy    =   position[0] + move[0]
        newx    =   position[1] + move[1]
        newPos  =   [max(0, min(newy, self.maze_dims[0]-1)), max(0, min(newx, self.maze_dims[1]-1))]
        # Compute reward
        mazeOver=   False
        wentOut =   newy<0 or newy>=self.maze_dims[0] or newx<0 or newx>=self.maze_dims[1] or self.maze_allowed[newy,newx]==0
        if wentOut:
            reward  =   self.move_reward[1]
        elif newPos in self.maze_finish:
            reward  =   self.move_reward[2]
            mazeOver=   True
        else:
            reward  =   self.move_reward[0]
        return reward, newPos, mazeOver

    # DISPLAY FUNCTIONS
    # =================
    def display(self, draw=False, videoTape=None):

        def update_matrix(num, dum, hndl):
            hndl.set_data(self.view_race(num))
            return hndl,

        def update_ticks():
            # Minor ticks - Ax1
            for xA in [self.ax1, self.ax2, self.ax3]:
                xA.set_xticks(np.arange(-.5, self.maze_dims[1], 1), minor=True)
                xA.set_yticks(np.arange(-.5, self.maze_dims[0], 1), minor=True)
                xA.grid(which='minor', color='k', linestyle='-', linewidth=1)

        # Init display
        nAgents =   len(self.agents)
        count   =   0
        if self.figId is None:
            # CREATE FIGURE
            self.figId  =   plt.figure()
            self.ax1    =   self.figId.add_subplot(100 + (2 + nAgents) * 10 + 1)
            self.ax2    =   self.figId.add_subplot(100 + (2 + nAgents) * 10 + 2)
            self.ax3    =   self.figId.add_subplot(100 + (2 + nAgents) * 10 + 3)
            # DRAW TRACK
            m1  =   np.zeros(np.shape(self.maze_reward))
            m1  =   ( ( self.maze_reward == 0 )  + 1 )/ 2 - 1 + abs(self.maze_allowed)
            self.imagePanels.append( self.ax1.imshow( np.dstack( [m1]*3 ), interpolation='nearest') )
            self.imagePanels.append( self.ax2.imshow( np.dstack( [m1]*3 ), interpolation='nearest') )
            self.imagePanels.append( self.ax3.imshow( np.dstack( [m1]*3 ), interpolation='nearest') )
        update_ticks()

        if not videoTape is None:
            # Initiate writer
            Writer      =   animation.writers['ffmpeg']
            writer      =   Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
            line_ani    =   animation.FuncAnimation(self.figId, update_matrix, max([len(x.position_chain) for x in self.racers]), fargs=([], self.imagePanels[0]),blit=False)
            line_ani.save(videoTape, writer=writer)
        else:
            self.imagePanels[0].set_data(self.view_race(-1))
            self.view_policy(-1)
            self.view_value(-1)
            plt.show()
            plt.draw()
            #plt.pause(0.1)

    def view_race(self, cnt):
        # ===========
        # TRACK MASKS
        # Mask1: raceTrack
        mask1   =   abs(self.maze_allowed)
        # Mask2: starting zone
        mask2   =   np.zeros(self.maze_dims)
        for iX, iY in self.maze_start:
            mask2[iX, iY] = .8
        # Mask3: Finish zone
        mask3 = np.zeros(self.maze_dims)
        for iX, iY in self.maze_finish:
            mask3[iX, iY] = .4
        # Prep track
        IMtrack = mask1 + mask2 + mask3
        IMtrack = np.dstack([IMtrack] * 3)
        # ===========
        # RACERS DOTS
        nAgents = len(self.agents)
        lsRGB   = [[1, 0, 0], [1, .5, 0], [1, 1, 0], [0.5, 1, 0], [0, 1, 0], [0, 1, 0.5], [0, 1, 1], [0, 0.5, 1],
                 [0, 0, 1], [0.5, 0, 1], [1, 0, 1], [1, 0, 0.5]]
        for iRac in range(nAgents):
            y, x = self.agents[iRac].position_chain[cnt]
            IMtrack[y, x, :] = lsRGB[iRac]
        return IMtrack

    def view_policy(self, cnt):
        # --- Draw the action arrows
        # Slice the policy
        for iy in range(self.maze_dims[0]):
            for ix in range(self.maze_dims[1]):
                pSlice  =   self.agents[0].policy[iy,ix,:]
                # Compute resultant vectors along each dimension
                resV    =   np.argmax(pSlice)
                ampV    =   0
                if sum(pSlice)>0:
                    ampV=   pSlice[resV]/sum(pSlice)
                # Draw arrows
                try:
                    self.arrowsP[iy][ix][0].remove()
                except:
                    pass
                iAct    =   self.agents[0].actions[resV]
                self.arrowsP[iy][ix] =   [self.ax2.arrow(-iAct[1]/2+ix, -iAct[0]/2+iy, iAct[1]/2, iAct[0]/2, head_width=0.5*ampV, head_length=max(max(abs(np.array(iAct)))/2, 0.001)*ampV, fc='k', ec='k')]

    def view_value(self, cnt):
        # --- Draw the action arrows
        for iy in range(self.maze_dims[0]):
            for ix in range(self.maze_dims[1]):
                pSlice  =   self.agents[0].global_value[iy, ix, :]
                # Compute resultant vectors along each dimension
                indV    =   [ np.multiply(x,y) for x,y in zip(pSlice, self.agents[0].actions)]
                resV    =   np.sum( np.array(indV), axis=0 )
                scl     =   np.sum( abs(np.array(indV)), axis=0)
                scl     =   [1 if x==0 else x for x in scl]
                resV    =   np.divide( resV, scl )
                ampV    =   np.sqrt( np.sum(resV**2) )
                # Draw arrows
                try:
                    self.arrowsV[iy][ix][0].remove()
                except:
                    pass
                self.arrowsV[iy][ix] = [self.ax3.arrow(-resV[1] / 2 + ix, -resV[0] / 2 + iy, resV[1] / 2, resV[0] / 2,
                                                       head_width=0.5 * ampV, head_length=max( ampV/ 2, 0.1), fc='k', ec='k')]



MZ  =   maze(type='template1', params=(0,0,1))


