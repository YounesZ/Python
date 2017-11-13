import numpy as np
from math import sin, cos, pi
from random import choice
from bisect import bisect_left
from sys import setrecursionlimit
from copy import deepcopy

class maze_agent():

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
        self.policy         =   np.zeros(mazeRunner.stateSpace)
        self.global_value   =   np.zeros(mazeRunner.stateSpace)
        self.local_value    =   {}
        self.planningModel  =   [[]] * mazeRunner.stateSpace[-1]
        self.planningiModel =   []
        for iDim in mazeRunner.stateSpace[:-1]:
            self.planningModel  =   [self.planningModel] * iDim
            self.planningiModel =   [self.planningiModel] * iDim

    def agent_move(self, moveSequence):
        # --- Here the agent receives a move signal and picks a move according to policy
        curS        =   self.mazeRunner.runnner_queryState()
        if len(moveSequence) > 0:
            # Replay a sequence of moves
            move    =   moveSequence.pop(0)
        else:
            # Select move according to policy
            moveP   =   eval( 'self.global_value[' + str(curS)[1:-1] + ']' )
            moveP   =   np.multiply(moveP, np.random.random([1, len(moveP)]))
            # Most probable move
            idProb  =   np.where( moveP==max(moveP[self.mazeRunner.actionsAllow]) )
            move    =   self.actions[ choice(idProb[0]) ]
        # Query next state
        nexS, rew, F=   self.mazeRunner.runner_change_velocity(self, move)
        # Learning phase
        self.agent_learn(curS, move, nexS, rew)
        # Planning phase
        self.agent_plan(curS, move, nexS, rew)
        # Update policy
        self.agent_updatePolicy(curS)
        if not F:
            self.agent_move(moveSequence)

    def agent_learn(self, prevState, prevAction, nextState, reward, incrementOnly=False):
        # Learn from experience
        Qprev       =   eval('self.global_value[' + str(prevState+prevAction)[1:-1] + ']')
        Qnext       =   eval( 'max( self.global_value[' + str(nextState)[1:-1] + '])' )
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
            self.planningModel[prevState][prevAction]  =   (reward, nexState)
            # Update the inverse model "which states drive me to nexState : prevState"
            self.planningiModel[nexState][tuple(prevState+prevAction)]      =   reward
            # Start backsearching tree
            S               =   prevState
            count           =   0
            # Insert previous state in priority queue "if worth it"
            while len(planningQueue) > 0 and count < self.planningNodes:
                # Empty the queue
                if len(planningPrior) > 0:
                    _       =   planningPrior.pop(-1)
                    S, A    =   planningQueue.pop(-1)
                    R, Sp   =   self.planningModel[S[0]][S[1]][A]
                    self.agent_learn(S, A, Sp, R)
                    # Update queue count
                    count   +=  1
                # States predicted to lead to S "This updates the queue"
                beforeS     =   self.planningiModel[S[0]][S[1]].keys()
                for bef in beforeS:
                    Sm, Am  =   list(bef[:2]), list(bef[2:])
                    Rm, _   =   self.planningModel[Sm[0]][Sm[1]][Am]
                    Priority=   abs(self.agent_learn(Sm, Am, S, Rm, incrementOnly=True))
                    if Priority > self.planningThresh and (Sm, Am) not in planningQueue:
                        # insertion position
                        ixS = bisect_left(planningPrior, Priority)
                        planningPrior.insert(ixS, Priority)
                        planningQueue.insert(ixS, (Sm, Am))

    def agent_updatePolicy(self, position):
        # Update the policy "find the most valuable move"
        moveP   =   self.global_value[position[0], position[1], :]
        # Find the action with highest value
        idProb  =   np.where(moveP == max(moveP))
        move    =   choice(idProb[0])
        # Update proba
        self.policy[position[0], position[1], :]    =   self.eGreedy / len(moveP)
        self.policy[position[0], position[1], move] =   1 - self.eGreedy * (1 - 1 / len(moveP))




# LAUNCHER
setrecursionlimit(10000)
from ReinforcementLearning.maze.maze import *

MZ = maze('raceTrack1')
# Link agent and maze
MA = maze_agent(MZ, planningNodes=0, planningThresh=0.1)
MZ.agents.append(MA)
MA.agent_restart()
"""
"""

"""
# EFFECT OF PLANNING THRESHOLD ON POLICY VALUES
# =============================================
# First make a path using non-planning agent
MZ      =   maze(display=False)
MA      = maze_agent(MZ, planningMode='noPlanning')
MZ.agents.append(MA)
MA.agent_restart()
mvSeq   =   MA.action_chain

# Make planning agents replay the path
FF      =   plt.figure()
axs     =   []
plTs    =   [0.1/(10**x) for x in range(5)]
imSh    =   [[] for x in range(5)]
imS     =   []
for ip, id in zip(plTs, range(len(plTs))):
    axs.append(FF.add_subplot(150 + id + 1))
for ip, id in zip(plTs, range(len(plTs))):
    MZ  =   maze(display=False)
    MA  =   maze_agent(MZ, planningNodes=10, planningThresh=ip)
    MZ.agents.append(MA)
    MA.agent_restart( moveSequence= deepcopy(mvSeq) )
    MA.agent_restart(moveSequence=deepcopy(mvSeq))
    imSh[id]    =   np.max(MA.global_value, axis=2)
    # Display
    imS.append( axs[id].imshow(imSh[id]) )
"""


"""
# EFFECT OF PLANNING NODES ON POLICY VALUES
# =========================================
# First make a path using non-planning agent
MZ      =   maze(display=False)
nNodes  =   range(11)
nRuns   =   9
nSteps  =   np.zeros( [len(nNodes), nRuns] )
nIter   =   25
for it in range(nIter):
    MA      =   []
    for iN in nNodes:
        MA.append( maze_agent(MZ, planningNodes=iN, planningThresh=0.001) )
    for iN in nNodes:
        for iRn in range(nRuns):
            MA[iN].agent_restart()
            nSteps[iN, iRn]     +=   (len(MA[iN].position_chain)-1)/nIter

# Display
FF      =   plt.figure()
axs     =   []
for iN in nNodes:
    axs.append( plt.plot( range(nRuns), nSteps[iN,:], label=str(iN)+' nodes') )
plt.legend()
plt.gca().set_ylim([10,800])
plt.gca().set_xlim([1, 8])
plt.gca().set_xlabel('Number of runs')
plt.gca().set_ylabel('Number of steps')
plt.title('Effect of planning on learning speed')
"""