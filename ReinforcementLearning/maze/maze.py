# This class implements various maze environments

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi, sin, cos
from copy import deepcopy
from random import choice
plt.ion()

class runner():

    # INIT FUNCTIONS
    # ==============
    def __init__(self, mazeInst, angleSection=0.25, maxVelocity=5, fieldOFview=(5,5)):
        # Actions
        angles              =   [x*pi*angleSection for x in range(0, int(2/angleSection))]
        self.actions        =   [[-round(sin(x)), round(cos(x))] for x in angles] + [[0,0]]
        self.actionsAllow   =   [True]*int(2/angleSection) + [False]
        self.environment    =   mazeInst
        # Velocities
        velX                =   list(range(-self.maxVelocity, self.maxVelocity + 1)) * (2 * self.maxVelocity + 1)
        velY                =   list(np.concatenate([[x] * (2 * self.maxVelocity + 1) for x in range(-self.maxVelocity, self.maxVelocity + 1)]))
        self.velocities     =   [[velY[idx], velX[idx]] for idx in range(len(velX))]
        self.maxVelocity    =   maxVelocity
        # Field of view
        self.viewY          =   fieldOFview[0]
        self.viewX          =   fieldOFview[1]
        # Space dimensions for learning
        self.state_space    =   mazeInst.maze_dims + [len(self.velocities)]
        self.action_space   =   len(self.actions)

    def runner_new_run(self, position, velocity=[0,0], FoV=None):
        # Empty learning variables
        self.state_chain    =   [(position, velocity)]
        self.action_chain   =   []
        self.cumul_reward   =   0
        self.cumul_steps    =   0

    def runner_change_velocity(self, acceleration):
        # Pick action: stochastic
        position    =   self.state_chain[-1][0]
        curVel      =   self.state_chain[-1][1]
        iAction     =   self.actions[acceleration]
        # Set car acceleration/deceleration
        velocity    =   np.add(self.velocities[curVel], iAction)  # UNCOMMENT THIS LINE FOR QUITTING SPEED-AND-STOP MODE
        velocity    =   [max(min(velocity[0], self.maxVelocity), -self.maxVelocity), max(min(velocity[1], self.maxVelocity), -self.maxVelocity)]
        newVel      =   self.velocities.index(velocity)
        # Get new position and velocities
        reward, newPos, gameOver    =   self.environment.compute_displacement(position, velocity)
        # Keep new variables
        self.action_chain.append(acceleration)
        self.state_chain.append((newPos, newVel))
        # Convert state-space to indices
        stateSpace  =   [position+[curVel], newPos+[newVel]]
        # Update allowed actions
        self.actionsAllow   =   [sum(abs(np.add(x, velocity))) > 0 for x in self.actions]
        return stateSpace, reward, gameOver

class maze():

    # INIT FUNCTIONS
    # ==============
    def __init__(self, type='maze1', params=(0,0,1), display=False):
        # Make the maze
        redFlag =   self.maze_make(type, params)
        if redFlag:
            return
        # Display variables
        self.displayOn      =   display
        self.figId          =   None
        self.arrowsP        =   [[[] for x in range(self.maze_dims[1])] for y in range(self.maze_dims[0])]
        self.arrowsV        =   [[[] for x in range(self.maze_dims[1])] for y in range(self.maze_dims[0])]
        # Agents variables
        self.agents         =   []

    def maze_make(self, type, params):
        # Common variables
        onMaze_rew  =   params[0]       # Penalty for transitions
        offMaze_rew =   params[1]       # Penalty for trying to quit maze
        finish_rew  =   params[2]       # Reward for finishing the maze

        # Maze 1:   Sutton and Barto, p.191
        if type=='maze1':
            h, w    =   6, 9
            start   =   [[2,0]]
            finish  =   [[0,8]]
            obstacle=   [(1,2,2,0), (4,5,0,0), (0,7,2,0)]
        elif type=='raceTrack1':
            h, w    =   32, 17
            start   =   [ [h-1,x] for x in range(3,9) ]
            finish  =   [ [x,w-1] for x in range(0,6) ]
            obstacle=   [(0,0,2,1), (0,2,0,0), (3,0,0,0), (14,0,7,0), (22,0,6,1), (29,0,2,2), (6,10,25,6), (7,9,24,0)]
        else:
            print('\n\nError: unrecognized track type.\n\n')
            return True

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
            newPos  =   position
        elif newPos in self.maze_finish:
            reward  =   self.move_reward[2]
            mazeOver=   True
        else:
            reward  =   self.move_reward[0]
        if self.displayOn:
            self.display()
        return reward, newPos, mazeOver

    # DISPLAY FUNCTIONS
    # =================
    def display(self, videoTape=None):

        def update_matrix(num, dum, hndl):
            hndl.set_data(self.view_race(num))
            return hndl,

        def update_ticks():
            # Minor ticks - Ax1
            self.ax1.set_xticks(np.arange(-.5, self.maze_dims[1], 1), minor=True)
            self.ax1.set_yticks(np.arange(-.5, self.maze_dims[0], 1), minor=True)
            self.ax1.grid(which='minor', color='k', linestyle='-', linewidth=1)

        # Init display
        nAgents =   len(self.agents)
        if self.figId is None:
            # CREATE FIGURE
            self.figId  =   plt.figure()
            self.ax1    =   self.figId.add_subplot(111);
            self.ax1.title.set_text('Run')

            # DRAW TRACK
            m1               =   ( ( self.maze_reward == 0 )  + 1 )/ 2 - 1 + abs(self.maze_allowed)
            self.imagePanels =   self.ax1.imshow( np.dstack( [m1]*3 ), interpolation='nearest')
        update_ticks()

        if not videoTape is None:
            # Initiate writer
            Writer      =   animation.writers['ffmpeg']
            writer      =   Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
            line_ani    =   animation.FuncAnimation(self.figId, update_matrix, max([len(x.position_chain) for x in self.racers]), fargs=([], self.imagePanels[0]),blit=False)
            line_ani.save(videoTape, writer=writer)
        else:
            self.imagePanels.set_data(self.view_race(-1))
            plt.show()
            plt.draw()
            plt.pause(0.1)

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
        mask3   =   np.zeros(self.maze_dims)
        for iX, iY in self.maze_finish:
            mask3[iX, iY] = .4
        # Prep track
        IMtrack =   mask1 + mask2 + mask3
        IMtrack =   np.dstack([IMtrack] * 3)
        # ===========
        # RACERS DOTS
        nAgents =   len(self.agents)
        lsRGB   =   [[1, 0, 0], [1, .5, 0], [1, 1, 0], [0.5, 1, 0], [0, 1, 0], [0, 1, 0.5], [0, 1, 1], [0, 0.5, 1],
                    [0, 0, 1], [0.5, 0, 1], [1, 0, 1], [1, 0, 0.5]]
        for iRac in range(nAgents):
            y, x=   self.agents[iRac].position_chain[cnt]
            IMtrack[y, x, :] = lsRGB[iRac]
        return IMtrack



