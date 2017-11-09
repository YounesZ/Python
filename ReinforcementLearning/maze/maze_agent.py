import numpy as np
from math import sin, cos, pi
from random import choice
from bisect import bisect_left

class maze_agent():

    def __init__(self, mazeInst, Lambda=0, learnRate=0.15, eGreedy=0.1, discount=0.8, planningMode='prioritized', planningNodes=0, navMode='global'):
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
        self.planningThresh =   0.05  # Minimum absolute increment for entering queue
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

    def agent_restart(self):
        # --- Here the agent just picks a starting position in the maze and make first move
        # Flush variables
        self.position_chain     =   []
        self.action_chain       =   []
        # Pick a starting position
        self.agent_move( choice(self.environment.maze_start) )

    def agent_move(self, position):
        # --- Here the agent receives a move signal and picks a move according to policy
        # Select move according to policy
        moveP       =   self.global_value[position[0], position[1], :]
        # Most probable move
        idProb      =   np.where( moveP==max(moveP) )
        move        =   self.actions[ choice(idProb[0]) ]
        self.action_chain.append(move)
        # Query next position
        rew, nex, F =   self.environment.compute_displacement(position, move)
        # Learning phase
        self.agent_learn(position, move, nex, rew)
        # Planning phase
        self.agent_plan(position, move, nex, rew)
        # Update policy
        self.agent_updatePolicy(position)
        # Store variables
        self.position_chain.append( position )
        self.action_chain.append( move )
        if not F:
            self.agent_move(nex)
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
            # Compute the priority of this backup
            Priority    =   abs( self.agent_learn(prevState, prevAction, nexState, reward, incrementOnly=True) )
            # Insert previous state in priority queue "if worth it"
            if Priority>self.planningThresh:
                # insertion position
                ixS     =   bisect_left(planningQueue, Priority)
                planningPrior.insert(ixS, Priority)
                planningQueue.insert(ixS, (prevState, prevAction))
            # Loop over elements in the queue: go backward in the maze
            while len(planningQueue)>0:
                # First state to backup
                S, A    =   planningQueue.pop(0)
                R, Sp   =   self.planningModel[S[0]][S[1]][self.actions.index(A)]
                self.agent_learn(S, A, Sp, R)
                # States predicted to lead to S
                beforeS =   self.planningiModel[S[0]][S[1]].keys()
                for bef in beforeS:
                    Sm, Am  =   list(bef[:2]), self.actions.index(list(bef[2:]))
                    Rm, _   =   self.planningModel[Sm[0]][Sm[1]][Am]
                    Priority=   abs(self.agent_learn(prevState, prevAction, nexState, reward, incrementOnly=True))
                    if Priority > self.planningThresh:
                        # insertion position
                        ixS = bisect_left(planningQueue, Priority)
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
from ReinforcementLearning.maze.maze import *
MZ = maze()
MZ = maze()

MA = maze_agent(MZ, planningNodes=10)
MA.agent_restart()
