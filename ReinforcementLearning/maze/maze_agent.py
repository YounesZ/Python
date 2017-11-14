import numpy as np
from math import sin, cos, pi
from random import choice
from bisect import bisect_left
from sys import setrecursionlimit
from copy import deepcopy
from ReinforcementLearning.maze.maze import *
setrecursionlimit(10000)


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
        # Space dimensions for learning
        self.policy         =   {}
        self.global_value   =   {}
        self.local_value    =   {}
        self.planningModel  =   {}
        self.planningiModel =   {}
        # Display variables
        self.figId          =   None
        self.ax             =   None


    def agent_move(self, moveSequence=[]):
        # --- Here the agent receives a move signal and picks a move according to policy
        curS        =   tuple( self.environment.runner_query_state() )
        # Select move according to policy
        if not curS in self.global_value:
            self.global_value[curS] =   np.zeros(self.environment.action_space)
            self.policy[curS]       =   np.zeros(self.environment.action_space)
        if len(moveSequence) > 0:
            # Replay a sequence of moves
            move    =   moveSequence.pop(0)
        else:
            # Select move according to policy
            moveP   =   self.policy[curS]
            moveP   =   np.multiply(moveP, np.random.random([1, len(moveP)]))
            # Most probable move
            idProb  =   np.where( [x and y for x,y in zip( (moveP==max(moveP[0][self.environment.actionsAllow]))[0], self.environment.actionsAllow)] )
            move    =   choice(idProb[0])
        # Query next state
        nexS, rew, F=   self.environment.runner_change_velocity(move)
        # Learning phase
        self.agent_learn(curS, move, tuple(nexS), rew)
        # Planning phase
        self.agent_plan(curS, move, tuple(nexS), rew)
        # Update policy
        self.agent_updatePolicy(curS)
        if not F:
            self.agent_move(moveSequence)

    def agent_learn(self, prevState, prevAction, nextState, reward, incrementOnly=False):
        # Learn from experience
        Qprev   =   self.global_value[prevState][prevAction]
        # If next state is unknown, intialize it
        Qnext   =   0
        if nextState in self.global_value.keys():
            Qnext   =   max( self.global_value[nextState] )
        increment   =   self.learnRate * (reward + self.discount*Qnext - Qprev)
        if incrementOnly:
            return increment
        else:
            self.global_value[prevState][prevAction]    +=  increment

    def agent_plan(self, prevState, prevAction, nexState, reward):
        # Select planning type
        if self.planningMode=='prioritized':
            # Initialize
            planningQueue   =   [(prevState, prevAction)]
            planningPrior   =   [reward]
            # Update the model "which state do I end up in from prevState : nexState"
            self.planningModel[prevState, prevAction]   = (reward, nexState)
            # Update the inverse model "which states drive me to nexState : prevState"
            if nexState not in self.planningiModel.keys():
                self.planningiModel[nexState]   =   {}
            self.planningiModel[nexState][prevState, (prevAction)]      =   reward
            # Start backsearching tree
            S               =   prevState
            count           =   0
            # Insert previous state in priority queue "if worth it"
            while len(planningQueue) > 0 and count < self.planningNodes and planningPrior[-1]>self.planningThresh:
                # Empty the queue
                if len(planningPrior) > 0:
                    _       =   planningPrior.pop(-1)
                    S, A    =   planningQueue.pop(-1)
                    R, Sp   =   self.planningModel[S, (A)]
                    self.agent_learn(S, A, Sp, R)
                    # Update queue count
                    count   +=  1
                # States predicted to lead to S "This updates the queue"
                beforeS     =   []
                if S in self.planningiModel.keys():
                    beforeS =   self.planningiModel[S].keys()
                for Sm, Am in beforeS:
                    Rm, _   =   self.planningModel[Sm, (Am)]
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
        policy          =   np.ones(self.environment.action_space) * self.eGreedy / len(moveP)
        policy[move]    =   1 - self.eGreedy * (1 - 1 / len(moveP))
        self.policy[position]   =   policy


"""
# LAUNCHER - TEST
# ===============
# Set the environment
MZ  =   Maze('maze1')
MZ.maze_add_runner(1, angleSection=0.5, maxVelocity=1)

# Set the agent
MA  =   [maze_agent(x) for x in MZ.Runners]

# Start race
[x.agent_move() for x in MA]
"""



# EFFECT OF PLANNING THRESHOLD ON POLICY VALUES
# =============================================
# First make a path using non-planning agent

"""
MZ  =   Maze('maze1', display=False)
MZ.maze_add_runner(1, angleSection=0.5, maxVelocity=1)

MA      =   maze_agent(MZ.Runners[0], planningMode='noPlanning')
MA.agent_move()
mvSeq   =   MA.environment.action_chain

imSlice =   np.zeros( MA.environment.state_space[:2] )
for ik in MA.global_value.keys():
    imSlice[ik[0], ik[1]]   +=  max(MA.global_value[ik])
plt.figure(); plt.imshow(imSlice)

# Make planning agents replay the path
FF      =   plt.figure()
plTs    =   [0.1/(10**x) for x in range(5)]
axs     =   [[] for x in range(len(plTs))]
imSh    =   [[] for x in range(len(plTs))]
imS     =   [[] for x in range(len(plTs))]
for id in range(len(plTs)):
    axs[id]     =   FF.add_subplot(150 + id + 1)
# Make move
for ip, id in zip(plTs, range(len(plTs))):
    axs[id]     =   FF.add_subplot(150 + id + 1)
    MZ  =   Maze('maze1', display=False)
    MZ.maze_add_runner(1, angleSection=0.5, maxVelocity=1)
    MA  =   maze_agent(MZ.Runners[0], planningNodes=20, planningThresh=ip, planningMode='prioritized')
    # Make move
    MA.agent_move( deepcopy(mvSeq) )
    MZ.Runners[0].runner_new_run( MZ.maze_start[0] )
    MA.agent_move(deepcopy(mvSeq))
    MZ.Runners[0].runner_new_run(MZ.maze_start[0])
    MA.agent_move(deepcopy(mvSeq))
    # Display aftermove
    imSlice     =   np.zeros( MA.environment.state_space[:2] )
    for ik in MA.global_value.keys():
        imSlice[ik[0], ik[1]]   +=  max(MA.global_value[ik])
    imSh[id]    =   imSlice
    imS[id]     =   axs[id].imshow(imSh[id])

"""


"""
# EFFECT OF PLANNING NODES ON POLICY VALUES
# =========================================
# First make a path using non-planning agent
nNodes  =   range(0,13,4)
nRuns   =   50
nSteps  =   np.zeros( [len(nNodes), nRuns] )
nIter   =   10
for it in range(nIter):
    MA  =   []
    MZ  = Maze(type='raceTrack1', display=False)
    for iN in nNodes:
        MZ.maze_add_runner(1, angleSection=0.5, maxVelocity=1)
        [MA.append( Maze_agent(x, planningNodes=iN, planningThresh=0.0001) ) for x in MZ.Runners]
    for iN in range(len(nNodes)):
        for iRn in range(nRuns):
            MA[iN].agent_move()
            nSteps[iN, iRn]     +=   (len(MA[iN].environment.state_chain)-1)/nIter
            MA[iN].environment.reset()

# Display
FF      =   plt.figure()
axs     =   []
for iN in range(len(nNodes)):
    axs.append( plt.plot( range(nRuns), nSteps[iN,:], label=str(iN)+' nodes') )
plt.legend()
plt.gca().set_ylim([np.min(nSteps),np.max(nSteps)])
plt.gca().set_xlim([0, nRuns-1])
plt.gca().set_xlabel('Number of runs')
plt.gca().set_ylabel('Number of steps')
plt.title('Effect of planning on learning speed')
"""
