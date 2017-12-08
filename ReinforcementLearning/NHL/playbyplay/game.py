import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Game:

    def __init__(self, dataRep, season, gameId=None, gameQty=None):
        # Retrieve game info
        dataPath    =   dataRep+'/Season_'+str(season)+'/playbyplay_'+str(season)+'.csv'
        dataFrame   =   pd.read_csv( dataPath, engine='python' )
        # Retrieve roster info
        rosterPath  =   dataRep+'/Season_'+str(season)+'/roster_'+str(season)+'.csv'
        rosterFrame =   pd.read_csv( rosterPath, engine='python' )
        # Make sure to pick right season
        dataFrame   =   dataFrame[ dataFrame.loc[:, 'season']==int(season)]
        # Store frames
        self.hd     =   list(dataFrame.dtypes.index)
        self.df     =   dataFrame
        self.df_wc  =   dataFrame       #Working copy
        self.rf     =   rosterFrame
        # Fecth line shifts
        self.lineShifts     =   {}


    def get_game_ids(self):
        # List all game numbers
        gNums   =   np.unique(self.df['gcode'])
        return gNums


    def pick_equalstrength(self):
        dataFrame   =   self.lineShifts
        # Filter out powerplays
        isEqs       =   dataFrame['equalstrength']
        self.lineShifts =   {_key:np.array(dataFrame[_key])[np.array(isEqs)] for _key in dataFrame.keys()}


    def pick_regulartime(self):
        dataFrame   =   self.lineShifts
        # Filter out overtime
        isRt        =   dataFrame['regulartime']
        self.lineShifts =   {_key:np.array(dataFrame[_key])[np.array(isRt)] for _key in dataFrame.keys()}


    def pick_game(self, gameId=None, gameQty=None):
        dataFrame   =   self.df
        # Filter game
        if not gameQty is None:
            # List all game numbers
            gNums   =   self.get_game_ids()
            gNums   =   np.where(dataFrame['gcode'] == gNums)[0][0] - 1
            dataFrame   =   dataFrame.loc[:gNums]
        if not gameId is None:
            # Keep only gameId
            dataFrame   =   dataFrame[dataFrame.loc[:, 'gcode'] == int(gameId)]
        # Store output
        self.df_wc  =   dataFrame


    def pull_offensive_players(self, dfRow, team='h'):
        # Get player IDs
        pID     =   [dfRow[team+str(x)] for x in range(1,7)]
        # Check positions
        pPOS    =   [self.rf.loc[self.rf['player.id']==x, 'pos'] for x in pID]
        pOFF    =   [(x.values[0]=='R' or x.values[0]=='L' or x.values[0]=='C') for x in pPOS]
        return (list( np.array(pID)[pOFF] )+[1,1,1])[:3]


    def pull_line_shifts(self, team='home'):
        # Pick the right team
        tmDict  =   {'home':'h', 'away':'a', 'both':'ha'}
        tmP     =   tmDict[team]

        # Make containers
        LINES       =   {'playersID':[], 'onice':[0], 'office':[], 'iceduration':[], 'SHOT':[0], 'GOAL':[0], 'equalstrength':[True], 'regulartime':[True]}
        # Loop on all table entries
        prevDt      =   []
        prevLine    =   np.array([1, 1, 1])
        if team=='both':
            prevLine=   (np.ones([1,3])[0], np.ones([1,3])[0])
        for idL, Line in self.df_wc.iterrows():
            if team=='both':
                curLine =   ( np.sort(self.pull_offensive_players(Line, 'h')), np.sort(self.pull_offensive_players(Line, 'a')) )
            else:
                curLine =   np.sort(self.pull_offensive_players(Line, tmP))

            # team of interest has changed?
            if len(prevDt)==0:
                prevDt  =   Line
                thch    =   False
            elif team=='both':
                thch    =   not (prevLine[0] == curLine[0]).all() or not (prevLine[1] == curLine[1]).all()
            else:
                thch    =   not (prevLine==curLine).all()

            if thch:
                # Terminate this shift
                LINES['playersID'].append(prevLine)
                LINES['office'].append(prevDt['seconds'])
                LINES['iceduration'].append(LINES['office'][-1] - LINES['onice'][-1])
                # Start new shift
                LINES['onice'].append(prevDt['seconds'])
                LINES['equalstrength'].append(prevDt['away.skaters']==6 and prevDt['home.skaters']==6)
                LINES['regulartime'].append(prevDt['period']<4)
                LINES['SHOT'].append(0)
                LINES['GOAL'].append(0)
            if Line['etype']=='GOAL' or Line['etype']=='SHOT':
                LINES[Line['etype']][-1]    +=  1
            if Line['etype']=='PENL':
                LINES['equalstrength'][-1]  =   False

            prevDt      =   Line
            prevLine    =   curLine

        # Terminate line history
        LINES['office'].append(Line['seconds'])
        LINES['iceduration'].append(LINES['office'][-1] - LINES['onice'][-1])
        LINES['playersID'].append(prevLine)
        # Store
        self.lineShifts =   LINES


"""
    class Line:


    class Event:


    class Player:
"""



# LAUNCHER
# ========
# Pointers
dataRep =   '/home/younesz/Documents/Databases/Hockey/PlayByPlay'
season  =   '20022003'
# Instantiate class
gc      =   Game(dataRep, season)    #20128, 20129, 20130, 20131, 20132, 20133, 20136, 20137, 20138, 20139, 20140, 20141]
gameId  =   gc.get_game_ids()
# Retrieve line shifts for each game
LS      =   []
for ig in gameId:
    gc.pick_game(gameId[0])
    gc.pull_line_shifts('both')
    gc.pick_equalstrength()
    gc.pick_regulartime()
    LS  +=  list( gc.lineShifts['iceduration'] )

"""

[x.pull_line_shifts() for x in gc]
[x.filter_line_shifts() for x in gc]
"""



"""
# VALIDATION 1
# ============
# Objective: check what is a meaningful duration of lines opposition on the ice
# - Plot line change curves to compare both teams behaviour - exemplar
LSa     =   []; [np.concatenate(LSa, x.lineShifts['away'], axis=1) for x in gC]
LSh     =   []; [np.concatenate(LSh, x.lineShifts['home'], axis=1) for x in gC]
LSb     =   []; [np.concatenate(LSb, x.lineShifts['both'], axis=1) for x in gC]
plt.figure();   plt.plot( LSh['iceon'][1:], np.ones(1,len(LSh['iceon'])-1), label='home' )
                plt.plot( LSa['iceon'][1:], np.ones(1,len(LSa['iceon'])-1), label='away' )
                plt.plot( LSb['iceon'][1:], np.ones(1,len(LSb['iceon'])-1), label='any' )
# - Plot the histogram of shift durations to set a threshold
plt.figure();   ax1 = plt.add_subplot(131); plt.hist()
"""

