import numpy as np
from math import sin, cos, pi
from random import choice
from bisect import bisect_left
from sys import setrecursionlimit
from copy import deepcopy
from ReinforcementLearning.maze.maze import *
setrecursionlimit(20000)


class Maze_agent():

    def __init__(self, mazeRunner, Lambda=0, learnRate=0.15, eGreedy=0.1, discount=0.8, planningMode='prioritized', planningNodes=0, planningThresh=0.05, navMode='global'):
        # ==============
        # Maze agent
        # Agent type
        self.Lambda         =   Lambda
        self.learnRate      =   learnRate
        self.eGreedy        =   eGreedy
        self.discount       =   discount
        self.planningMode   =   planningMode
        self.planningNodes  =   planningNodes
        self.navMode        =   navMode
        self.environment    =   mazeRunner
        self.planningThresh =   planningThresh  # Minimum absolute increment for entering queue
        self.exploratoryMove=   []
        # Space dimensions for learning
        self.policy         =   np.zeros( [self.environment.public_SS, self.environment.public_AS] )
        self.global_value   =   np.zeros( [self.environment.public_SS, self.environment.public_AS] )-5
        self.local_value    =   {}
        self.planningModel  =   [ [[] for x in range(self.environment.public_AS)] for y in range(self.environment.public_SS)]
        self.planningiModel =   [ {} for x in range(self.environment.public_SS) ]
        # Display variables
        self.figId          =   None
        self.ax             =   None

    def agent_move(self, moveSequence=[]):
        # --- Here the agent receives a move signal and picks a move according to policy
        F   =   False
        init=   True
        while not F:
            # Current state
            curS    =   self.environment.runner_query_state()
            # Pick action
            if len(moveSequence) > 0:
                # Replay a sequence of moves
                move    =   moveSequence.pop(0)
            else:
                # Select move according to policy
                moveP   =   self.policy[curS,:]
                moveP   =   np.multiply(moveP, np.random.random([1, len(moveP)]))
                if init:
                    moveP   =   np.random.random([1, len(self.policy[curS,:])])
                # Most probable move
                idProb  =   np.where( [x and y for x,y in zip( (moveP==max(moveP[0][self.environment.actionsAllow]))[0], self.environment.actionsAllow)] )
                move    =   choice(idProb[0])
                if move != np.argmax(self.policy[curS,:]):
                    self.exploratoryMove.append(1)
            # Query next state
            nexS, rew, F=   self.environment.runner_change_velocity(move)
            # Learning phase
            self.agent_learn(curS, move, nexS, rew)
            # Planning phase
            self.agent_plan(curS, move, nexS, rew)
            # Update policy
            self.agent_updatePolicy(curS)
            init    =   False

    def agent_learn(self, prevState, prevAction, nextState, reward, incrementOnly=False):
        # Learn from experience
        Qprev   =   self.global_value[prevState, prevAction]
        Qnext   =   max( self.global_value[nextState,:] )
        increment   =   self.learnRate * (reward + self.discount*Qnext - Qprev)
        if incrementOnly:
            return increment
        else:
            self.global_value[prevState, prevAction]    +=  increment

    def agent_plan(self, prevState, prevAction, nexState, reward):
        # Select planning type
        if self.planningMode=='prioritized':
            # Initialize
            planningQueue   =   [(prevState, prevAction)]
            planningPrior   =   [reward]
            # Update the model "which state do I end up in from prevState : nexState"
            self.planningModel[prevState][prevAction]   = (reward, nexState)
            # Update the inverse model "which states drive me to nexState : prevState"
            self.planningiModel[nexState][(prevState, prevAction)]      =   reward
            # Start backsearching tree
            S               =   prevState
            nodes           =   []
            # Insert previous state in priority queue "if worth it"
            while len(planningQueue) > 0 and len(nodes) < self.planningNodes and planningPrior[-1]>self.planningThresh:
                # Empty the queue
                if len(planningPrior) > 0:
                    _       =   planningPrior.pop(-1)
                    S, A    =   planningQueue.pop(-1)
                    R, Sp   =   self.planningModel[S][A]
                    self.agent_learn(S, A, Sp, R)
                    # Update queue count
                    if S not in nodes: nodes.append(S)
                # States predicted to lead to S "This updates the queue"
                beforeS     =   self.planningiModel[S].keys()
                for Sm, Am in beforeS:
                    Rm, _   =   self.planningModel[Sm][Am]
                    Priority=   abs(self.agent_learn(Sm, Am, S, Rm, incrementOnly=True))
                    if Priority > self.planningThresh and (Sm, Am) not in planningQueue:
                        # insertion position
                        ixS = bisect_left(planningPrior, Priority)
                        planningPrior.insert(ixS, Priority)
                        planningQueue.insert(ixS, (Sm, Am))

    def agent_updatePolicy(self, position):
        # Update the policy "find the most valuable move"
        moveP   =   self.global_value[position]
        # Find the action with highest value
        idProb  =   np.where(moveP == max(moveP))
        move    =   choice(idProb[0])
        # Update proba
        policy          =   np.ones(self.environment.public_AS) * self.eGreedy / len(moveP)
        policy[move]    =   1 - self.eGreedy * (1 - 1 / len(moveP))
        self.policy[position]   =   policy


    # DISPLAY PART
    # ============
    def display(self, index=None):

        self.arrowsP = [[[] for x in range(self.environment.environment.maze_dims[1])] for y in range(self.environment.environment.maze_dims[0])]
        self.arrowsV = [[[] for x in range(self.environment.environment.maze_dims[1])] for y in range(self.environment.environment.maze_dims[0])]

        def update_matrix(num, dum, hndl):
            hndl.set_data(self.view_race(num))
            return hndl,

        def update_ticks():
            # Minor ticks - Ax1
            for x in [self.ax1, self.ax2]:
                x.set_xticks(np.arange(-.5, self.environment.environment.maze_dims[1], 1), minor=True)
                x.set_yticks(np.arange(-.5, self.environment.environment.maze_dims[0], 1), minor=True)
                x.grid(which='minor', color='k', linestyle='-', linewidth=1)

        # Init display
        if self.figId is None:
            # CREATE FIGURE
            self.figId  =   plt.figure()
            self.ax1    =   self.figId.add_subplot(121);
            self.ax1.title.set_text('Policy');
            self.imagePanels    =   [ self.ax1.imshow( self.environment.environment.view_race(-1) ) ]
            self.ax2    =   self.figId.add_subplot(122);
            self.ax2.title.set_text('Action value');
            self.imagePanels.append( self.ax2.imshow( self.environment.environment.view_race(-1) ) )
        update_ticks()

        self.imagePanels[0].set_data( self.environment.environment.view_race(-1) );
        self.imagePanels[1].set_data( self.environment.environment.view_race(-1) );
        self.view_policy(index)
        self.view_value(index)
        plt.show()
        plt.draw()
        plt.pause(0.1)

    def view_policy(self, index):
        # --- Draw the action arrows
        # Slice the policy
        POL     =   np.reshape( self.policy, self.environment.private_SS+[self.environment.public_AS] )
        if not index is None:
            POL =   POL[:,:,index,:]

        DIMS    =   np.shape(POL)
        for iy in range(DIMS[0]):
            for ix in range(DIMS[1]):
                pSlice = POL[iy, ix, :]
                # Compute resultant vectors along each dimension
                resV = np.argmax(pSlice)
                ampV = 0
                if sum(pSlice) > 0:
                    ampV = pSlice[resV] / sum(pSlice)
                # Draw arrows
                try:
                    self.arrowsP[iy][ix][0].remove()
                except:
                    pass
                iAct = self.environment.actions[resV]
                self.arrowsP[iy][ix] = [
                    self.ax1.arrow(-iAct[1] / 2 + ix, -iAct[0] / 2 + iy, iAct[1] / 2, iAct[0] / 2,
                                   head_width=0.5 * ampV,
                                   head_length=max(max(abs(np.array(iAct))) / 2, 0.001) * ampV, fc='k', ec='k')]

    def view_value(self, index):
        # --- Draw the action arrows
        VAL = np.reshape(self.global_value, self.environment.private_SS + [self.environment.public_AS])
        if not index is None:
            VAL = VAL[:, :, index, :]

        DIMS = np.shape(VAL)
        for iy in range(DIMS[0]):
            for ix in range(DIMS[1]):
                pSlice  =   VAL[iy, ix, :]
                # Compute resultant vectors along each dimension
                indV    =   [np.multiply(x, y) for x, y in zip(pSlice, self.environment.actions)]
                resV    =   np.sum(np.array(indV), axis=0)
                scl     =   np.sum(abs(np.array(indV)), axis=0)
                scl     =   [1 if x == 0 else x for x in scl]
                resV    =   np.divide(resV, scl)
                ampV    =   np.sqrt(np.sum(resV ** 2))
                # Draw arrows
                try:
                    self.arrowsV[iy][ix][0].remove()
                except:
                    pass
                self.arrowsV[iy][ix] = [self.ax2.arrow(-resV[1] / 2 + ix, -resV[0] / 2 + iy, resV[1] / 2, resV[0] / 2,
                                                       head_width=0.5 * ampV, head_length=max(ampV / 2, 0.1), fc='k',
                                                       ec='k')]


# LAUNCHER - TEST
# ===============
# Set the environment
"""
MZ  =   Maze('maze1')
MZ.maze_add_runner(1, angleSection=0.5, maxVelocity=1)

# Set the agent
MZ.display()
MA  =   [Maze_agent(x, planningMode='prioritized', planningThresh=0.00001, planningNodes=20) for x in MZ.Runners]


# Start race
[x.reset() for x in MZ.Runners]
[x.agent_move() for x in MA]
MA[0].display(4,None)

# re-Start race
[x.reset() for x in MZ.Runners]
[x.agent_move() for x in MA]
MA[0].display(None, 4)
"""

"""
# EFFECT OF PLANNING THRESHOLD ON POLICY VALUES
# =============================================
# First make a path using non-planning agent


MZ  =   Maze('maze1', display=False)
MZ.maze_add_runner(1, angleSection=0.5, maxVelocity=1)

MA      =   Maze_agent(MZ.Runners[0], planningMode='noPlanning')
MA.agent_move()
mvSeq   =   MA.environment.action_chain

imSlice =   np.mean( np.max( np.reshape( MA.global_value, MA.environment.private_SS+[MA.environment.public_AS] ), axis=3 ), axis=2 )
plt.figure(); plt.imshow(imSlice)

# Make planning agents replay the path
FF      =   plt.figure()
plTs    =   [0.1/(10**x) for x in range(5)]
axs     =   [[] for x in range(len(plTs))]
imSh    =   [[] for x in range(len(plTs))]
imS     =   [[] for x in range(len(plTs))]
MA      =   [Maze_agent(MZ.Runners[0], planningNodes=10, planningThresh=x, planningMode='prioritized') for x in plTs]
for id in range(len(plTs)):
    axs[id]     =   FF.add_subplot(150 + id + 1)
    # Make move
    MA[id].agent_move( deepcopy(mvSeq) )
    MA[id].environment.reset()
    # Display aftermove
    imSlice     =   np.reshape( MA[id].global_value, MA[id].environment.private_SS+[MA[id].environment.public_AS] )
    imSh[id]    =   np.max( imSlice[:,:,4,:], axis=2 )
    imS[id]     =   axs[id].imshow(imSh[id])

"""




# EFFECT OF PLANNING NODES ON POLICY VALUES
# =========================================
# First make a path using non-planning agent
nNodes  =   range(0,25,8)
nRuns   =   5000
nSteps  =   np.zeros( [len(nNodes), nRuns] )
nIter   =   25
# Make environment
MZ      =   Maze(type='raceTrack1', display=False, params=(-1,-5,5))
MZ.maze_add_runner( 1, angleSection=0.25, maxVelocity=5 )
# Make agents
for it in range(nIter):
    MA      =   [Maze_agent(MZ.Runners[0], eGreedy=0.1, planningNodes=x, planningThresh=0.0001, planningMode='prioritized') for x in nNodes]
    print('Iteration: '+str(it))
    for iRn in range(nRuns):
        for iN in range(len(nNodes)):
            MA[iN].agent_move()
            nSteps[iN, iRn] += (len(MA[iN].environment.state_chain) - 1) / nIter
            MA[iN].environment.reset()

# Display
FF      =   plt.figure()
axs     =   []
for iN in range(len(nNodes)):
    axs.append( plt.plot( range(nRuns), nSteps[iN,:], label=str(nNodes[iN])+' nodes') )
plt.legend()
plt.gca().set_ylim([np.min(nSteps),np.max(nSteps)])
plt.gca().set_xlim([0, nRuns-1])
plt.gca().set_xlabel('Number of runs')
plt.gca().set_ylabel('Number of steps')
plt.title('Effect of planning on learning speed')
"""
"""