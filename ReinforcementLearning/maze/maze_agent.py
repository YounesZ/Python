import numpy as np
from math import sin, cos, pi
from random import choice
from bisect import bisect_left
from sys import setrecursionlimit
from copy import deepcopy

class maze_agent():

    def __init__(self, mazeInst, Lambda=0, learnRate=0.15, eGreedy=0.1, discount=0.8, planningMode='prioritized', planningNodes=0, planningThresh=0.05, navMode='global'):
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
        self.environment    =   mazeInst
        self.planningThresh =   planningThresh  # Minimum absolute increment for entering queue
        # Actions
        angles              =   range(0,4)
        self.actions        =   [[-round(sin(x*pi/2)), round(cos(x*pi/2))] for x in angles]
        # Field of view
        self.viewX          =   5
        self.viewY          =   5
        # Space dimensions for learning
        self.policy         =   np.zeros(mazeInst.maze_dims + [len(self.actions)])
        self.global_value   =   np.zeros(mazeInst.maze_dims + [len(self.actions)])
        self.planningModel  =   [ [[[] for w in range(len(self.actions))] for y in range(mazeInst.maze_dims[1])] for z in range(mazeInst.maze_dims[0])]
        self.planningiModel =   [[{} for x in range(mazeInst.maze_dims[1])] for y in range(mazeInst.maze_dims[0])]
        self.local_value    =   {}

    def agent_restart(self, moveSequence=[]):
        # --- Here the agent just picks a starting position in the maze and make first move
        # Flush variables
        self.position_chain     =   []
        self.action_chain       =   []
        # Pick a starting position
        self.agent_move( choice(self.environment.maze_start), moveSequence )

    def agent_move(self, position, moveSequence):
        # --- Here the agent receives a move signal and picks a move according to policy
        if len(moveSequence) > 0:
            # Replay a sequence of moves
            move    =   moveSequence.pop(0)
        else:
            # Select move according to policy
            moveP   =   self.global_value[position[0], position[1], :]
            # Most probable move
            idProb  =   np.where( moveP==max(moveP) )
            move    =   self.actions[ choice(idProb[0]) ]
        # Store variables
        self.position_chain.append(position)
        self.action_chain.append(move)
        # Query next position
        rew, nex, F =   self.environment.compute_displacement(position, move)
        # Learning phase
        self.agent_learn(position, move, nex, rew)
        # Planning phase
        self.agent_plan(position, move, nex, rew)
        # Update policy
        self.agent_updatePolicy(position)
        if not F:
            self.agent_move(nex, moveSequence)
        else:
            self.position_chain.append(nex)

    def agent_learn(self, prevState, prevAction, nextState, reward, incrementOnly=False):
        # Learn from experience
        Qprev       =   self.global_value[prevState[0], prevState[1], self.actions.index(prevAction)]
        Qnext       =   max( self.global_value[nextState[0], nextState[1], :] )
        increment   =   self.learnRate * (reward + self.discount*Qnext - Qprev)
        if incrementOnly:
            return increment
        else:
            self.global_value[prevState[0], prevState[1], self.actions.index(prevAction)]   +=  increment

    def agent_plan(self, prevState, prevAction, nexState, reward):
        # Select planning type
        if self.planningMode=='prioritized':
            # Initialize
            planningQueue   =   []
            planningPrior   =   []
            # Update the model "which state do I end up in from prevState : nexState"
            self.planningModel[prevState[0]][prevState[1]][self.actions.index(prevAction)]  =   (reward, nexState)
            # Update the inverse model "which states drive me to nexState : prevState"
            self.planningiModel[nexState[0]][nexState[1]][tuple(prevState+prevAction)]      =   reward
            # Start backsearching tree
            S               =   prevState
            init            =   True
            # Insert previous state in priority queue "if worth it"
            while len(planningQueue) > 0 or init:
                # States predicted to lead to S "This updates the queue"
                init        =   False
                beforeS     =   self.planningiModel[S[0]][S[1]].keys()
                for bef in beforeS:
                    Sm, Am  =   list(bef[:2]), list(bef[2:])
                    Rm, _   =   self.planningModel[Sm[0]][Sm[1]][self.actions.index(Am)]
                    Priority=   abs(self.agent_learn(Sm, Am, S, Rm, incrementOnly=True))
                    if Priority > self.planningThresh and (Sm, Am) not in planningQueue:
                        # insertion position
                        ixS = bisect_left(planningPrior, Priority)
                        planningPrior.insert(ixS, Priority)
                        planningQueue.insert(ixS, (Sm, Am))
                planningQueue = planningQueue[:self.planningNodes]
                planningPrior = planningPrior[:self.planningNodes]
                # Empty the queue
                if len(planningPrior)>0:
                    _       =   planningPrior.pop(-1)
                    S, A    =   planningQueue.pop(-1)
                    R, Sp   =   self.planningModel[S[0]][S[1]][self.actions.index(A)]
                    self.agent_learn(S, A, Sp, R)

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

"""
MZ = maze(display=False)
# Link agent and maze
MA = maze_agent(MZ, planningNodes=10, planningThresh=0.1)
MZ.agents.append(MA)
MA.agent_restart()
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
for ip, id in zip(plTs, range(len(plTs))):
    axs.append(FF.add_subplot(150 + len(imSh)))
for ip, id in zip(plTs, range(len(plTs))):
    MZ  =   maze(display=False)
    MA  =   maze_agent(MZ, planningNodes=10, planningThresh=ip)
    MZ.agents.append(MA)
    MA.agent_restart( moveSequence= deepcopy(mvSeq) )
    imSh[id]    =   np.max(MA.global_value, axis=2)
    # Display
    axs[id].imshow(imSh[id])




