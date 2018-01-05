import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout
from copy import deepcopy
from Utils.programming.ut_find_folders import *
from ReinforcementLearning.NHL.playerstats.nhl_player_stats import pull_stats


class HockeySS:

    def __init__(self, repoPbP, repoPSt):
        self.repoPbP    =   repoPbP
        self.repoPSt    =   repoPSt
        self.seasons    =   ut_find_folders(repoPbP, True)


    def list_all_games(self):
        # List games
        games_lst       =   pd.DataFrame()
        for iy in self.seasons:
            iSea        =   Season(iy)
            iSea.list_game_ids( path.join(self.repoPbP, iSea) )
            games_lst   =   pd.concat( (games_lst, iSea.games_id), axis=0 )
        self.games_lst  =   games_lst


    def pull_line_data(self):
        # List line shifts
        shifts_lst      =   pd.DataFrame()
        count           =   0
        for iy,ic in zip(self.games_lst['season'].values,self.games_lst['gcode'].values):
            iGame       =   Game(self.repoPbP, self.repoPSt, iy, ic)
            iGame.pull_line_shifts('both')
            iGame.pull_player_categories()

            # Save data
            shifts_lst  =   pd.concat( (shifts_lst, pd.DataFrame.from_dict( iGame.lineShifts )), axis=0, ignore_index=True )

            # Status bar
            stdout.write('\r')
            # the exact output you're looking for:
            stdout.write("Game %i/%i - season %s game %s: [%-60s] %d%%, completed" % (count, len(self.games_lst), iy, ic, '=' * int(count / len(self.games_lst) * 60), 100 * count / len(self.games_lst)))
            stdout.flush()

            count   +=  1
        self.line_shifts=   shifts_lst


class Season:

    def __init__(self, year):
        self.year   =   year


    def list_game_ids(self, dataRep):
        # Format year
        iyear           =   self.year.replace('Season_', '')
        # Get data - long
        gc              =   Game(dataRep, iyear)
        # Get game IDs
        self.games_id   =   gc.df.drop_duplicates(subset=['season', 'gcode'], keep='first')[['season', 'gcode', 'refdate']]


class Game:

    def __init__(self, repoPbP, repoPSt, season, gameId=None, gameQty=None):
        # Retrieve game info
        self.season =   season
        self.gameId =   gameId
        self.repoPbP=   repoPbP
        self.repoPSt=   repoPSt
        dataPath    =   path.join(repoPbP, 'Season_'+str(season), 'converted_data.p')
        dataFrames  =   pickle.load( open(dataPath, 'rb') )
        # Make sure to pick right season
        #dataFrame   =   dataFrame[ dataFrame.loc[:, 'season']==int(season)]
        # Store frames
        self.hd     =   dataFrames['playbyplay'].columns
        self.df     =   dataFrames['playbyplay']
        self.df_wc  =   dataFrames['playbyplay']       #Working copy
        self.rf     =   dataFrames['roster']
        # Fecth line shifts
        self.lineShifts     =   {}
        # Filter for game Id
        if not gameId is None:
            self.df_wc  =   self.df[self.df['gcode']==gameId]


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
        LINES       =   {'playersID':[], 'onice':[0], 'office':[], 'iceduration':[], 'SHOT':[0], 'GOAL':[0], 'BLOCK':[0], 'MISS':[0], 'PENL':[0], 'equalstrength':[True], 'regulartime':[True]}
        # Loop on all table entries
        prevDt      =   []
        prevLine    =   np.array([1, 1, 1])
        evTypes     =   ['GOAL', 'SHOT', 'PENL', 'BLOCK', 'MISS']
        ts_a        =   0
        ts_h        =   0
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
                LINES['PENL'].append(0)
                LINES['BLOCK'].append(0)
                LINES['MISS'].append(0)
            if any([x==Line['etype'] for x in evTypes]):
                sign    =   int(Line['hometeam']==Line['ev.team'])*2-1
                LINES[Line['etype']][-1]    +=  sign
                if Line['etype']=='GOAL':
                    LINES['SHOT'][-1]       +=  sign
            if Line['etype']=='PENL':
                LINES['equalstrength'][-1]  =   False
            prevDt      =   deepcopy(Line)
            prevLine    =   deepcopy(curLine)

        # Terminate line history
        LINES['office'].append(Line['seconds'])
        LINES['iceduration'].append(LINES['office'][-1] - LINES['onice'][-1])
        LINES['playersID'].append(prevLine)
        # Store
        self.lineShifts =   LINES


    def pull_player_categories(self):
        # List concerned players
        all_pl  =   self.lineShifts['playersID'].values
        all_plC =   np.unique( np.concatenate(all_pl) )
        all_plN =   self.rf.set_index('Unnamed: 0').loc[all_plC[all_plC>1]]['firstlast']
        # Get raw player stats
        gcode   =   str(self.season)[:4]+'0'+str(self.gameId)
        all_plS =   pull_stats(self.repoPSt, self.repoPbP, uptocode=gcode, nGames=30, plNames=all_plN.values)




"""     
    class Line:


    class Event:


    class Player:
"""


# LAUNCHER
# ========
# Pointers
root        =   '/home/younesz/Documents'
#root        =   '/Users/younes_zerouali/Documents/Stradigi'
repoPbP     =   path.join(root, 'Databases/Hockey/PlayByPlay')
repoPSt     =   path.join(root, 'Databases/Hockey/PlayerStats/player')


HSS     =   HockeySS(repoPbP, repoPSt)
#HSS.list_all_games()
#HSS.pull_line_data()


"""
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

