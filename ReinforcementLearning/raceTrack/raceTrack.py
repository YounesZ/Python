""" This code solves the raceTrack problem in Sutton and Barto 5.6 """


import numpy as np



class raceTrack():

    def __init__(self, trackType=1):
        # Generate track shape
        self.track_init(trackType)

    def track_init(self, trackType):
        # First example
        if trackType==1:
            # Dimensions
            trHeight, trWidth   =   (32, 17)
            # Forbidden zones
            forbidden           =   [3,0]+[2,0]*2+[1,0]+[0,0]*2+[0,7]+[0,8]*7+[1,8]*8+[2,8]*7+[3,8]*3
            forbidden           =   [forbidden[(x*2):(x*2)+2] for x in range(int(len(forbidden)/2))]
            # Starting zone
            startZone           =   [list(range(4, 10)), [32]*6]
            # Finish zone
            finishZone          =   [[17]*6, list(range(1,7))]
        # Second example
        elif trackType==2:
            # Dimensions
            trHeight, trWidth   =   (30, 32)
            # Forbidden zones
            forbidden           =   [16, 0] + [13, 0] + [12,0] + [11, 0]*4 + [12,0] + [13,0] + [14, 2] + [14,5] + [14,6] + [14, 8] + [14, 9] + [13, 9] + [12, 9] + [11, 9] + [10, 9] + [9, 9] + [8, 9] + [7, 9] + [6, 9] + [5, 9] + [4, 9] + [3, 9] + [2, 9] + [1, 9] + [0, 9] + [0, 9] + [0, 9]
            forbidden           =   [forbidden[(x * 2):(x * 2) + 2] for x in range(int(len(forbidden) / 2))]
            # Starting zone
            startZone           =   [list(range(1, 24)), [30] * 23]
            # Finish zone
            finishZone          =   [[32] * 9, list(range(1, 10))]
        # Make track
        track_reward    =   [np.array([-5]*(x[0]+1)+[-1]*(trWidth-sum(x))+[-5]*(x[-1]+1)) for x in forbidden]
        track_reward    =   np.append( np.append( np.array([-5]*(trWidth+2), ndmin=2), track_reward, axis=0 ), np.array([-5]*(trWidth+2), ndmin=2), axis=0)
        # Make start
        for ii in range(np.shape(startZone)[-1]):
            track_reward[startZone[1][ii], startZone[0][ii]]    =   0
        # Make finish
        for ii in range(np.shape(finishZone)[-1]):
            track_reward[finishZone[1][ii], finishZone[0][ii]]  =   5
        # EOF
        self.track_dim      =   (trWidth, trHeight)
        self.track_reward   =   track_reward

    def compute_displacement(self, position, velocity):
        # This function issues the new state and reward after displacement
        newPos      =   position + velocity
        newPos[0]   =   max( min(newPos[0], 0), self.track_dim[0] )
        newPos[1]   =   max( min(newPos[1], 0), self.track_dim[1] )
        # Compute reward
        reward      =   self.track_reward[newPos[0], newPos[1]]
        decrement   =   [0, 0]
        loopon      =   reward<0
        while loopon:
            # New decrement
            decrement   +=  [velocity[0]/sum(velocity), velocity[1]/sum(velocity)]
            # Walk back
            newPos2     =   newPos - [int(decrement[0]), int(decrement[1])]
            # New reward
            nRew        =   self.track_reward[newPos2[0], newPos2[1]]
            loopon      =   nRew<0
            if not loopon:
                newPos  =   newPos2
