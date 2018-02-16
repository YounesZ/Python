import numpy as np

class Agent():

    def __init__(self, nStates, nActions, type='TDlambda', Lambda=0, learnRate=0.15, discount=0.8):
        # Agent type
        self.Lambda         =   Lambda
        self.learnRate      =   learnRate
        self.discount       =   discount
        # Initiate the learning variables
        self.action_value   =   np.zeros([nStates, nActions])
        self.agent_init()

    def agent_init(self):
        self.elig_trace     =   np.zeros( np.shape(self.action_value) )

    def agent_move(self, S, A, R, Sp):
        # Compute Q-values
        Qold    =   self.action_value[S, A]
        init    =   False
        if Sp==[]:
            Qnew=   Qold
            init=   True
        else:
            Qnew=   np.max( self.action_value[Sp,:] )
        # Update Q-value
        incr    =   R + self.discount * Qnew - Qold
        self.elig_trace[S, A]   =   1
        self.action_value       +=  self.learnRate * incr * self.elig_trace
        self.elig_trace         *=  self.discount * self.Lambda
        # Re-init eligibility trace
        if init:
            self.agent_init()

