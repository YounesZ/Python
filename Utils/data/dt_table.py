import numpy as np
import csv
import re
from os import getcwd, listdir, chdir, stat
from Utils.programming import ut_unique
from Utils.data.dt_tools import *

class Table:

    def __init__(self, fileLoc, fileName):
        # Data location
        self.fileLoc = fileLoc
        self.fileName = fileName

    def read(self):
        # ======= Read PLAY-BY-PLAY
        # Load file
        PbP = []
        with open(self.fileLoc + '/' + self.fileName, newline='') as csvfile:
            spamreader  =   csv.reader(csvfile, dialect='excel', delimiter='\t', quotechar='|')
            for row in spamreader:
                PbP.append(re.sub('"', '', row[0]).split(','))
        self.header     =   PbP[0]
        self.dataRaw    =   PbP[1:]

    def get_column_idx(self, colName):
        colId   =   self.header.index(colName)
        colDt   =   [x[colId] for x in self.dataRaw]
        return colId, colDt

    def filter_column(self, colId_target, colId_filter, filter):
        colFlt  =   []
        for ii in self.dataRaw:
            if ii[colId_filter]==filter:
                colFlt.append(ii[colId_target])
        return colFlt

    def get_column_range(self, colId, value):
        vRng    =   {'start':[], 'end':[]}
        prev    =   False
        count   =   0
        for ii in self.dataRaw:
            if ii[colId]==value and not(prev):
                vRng['start'].append(count)
            elif ii[colId]!=value and prev:
                vRng['end'].append(count-1)
            prev    =   ii[colId]==value
            count   +=  1
        return vRng

    def slice_table(self, range_row, range_col):
        slice   =   []
        for ii in range(range_row[0], range_row[1]+1):
            slice.append(self.dataRaw[ii][range_col[0]:range_col[1]+1])
        return slice


class PBPtable(Table):

    def __init__(self, fileLoc, fileName, plClasses=['C', 'L', 'R', 'D']):
        # Data location
        Table.__init__(self, fileLoc, fileName)
        self.plClasses  =   plClasses

    def detect_power_plays(self):
        # Find home player indices
        hpi     =   [self.get_column_idx('h'+str(x)) for x in [1,5]]
        hpi     =   [x[0] for x in hpi]
        # Find away player indices
        api     =   [self.get_column_idx('a'+str(x)) for x in [1,5]]
        api     =   [x[0] for x in api]
        # Loop and detect powerplays: 0=5on5, 1=homePplay, -1=awayPplay
        self.header +=  ['powerplay']
        for ii in range(len(self.dataRaw)):
            # Home players
            plH     =   self.dataRaw[ii][hpi[0]:hpi[1]+1]
            plH     =   5 - sum([x=='1' for x in plH])
            # Away players
            plA     =   self.dataRaw[ii][api[0]:api[1]+1]
            plA     =   5 - sum([x == '1' for x in plA])
            # classify play
            self.dataRaw[ii]    +=  [plH*10 + plA]



class ROSTERtable(Table):

    def __init__(self, fileLoc, fileName, plClasses=['C', 'L', 'R', 'D']):
        # Data location
        Table.__init__(self, fileLoc, fileName)
        self.plClasses  =   plClasses



# Read the tables
#REP     =   '/home/younesz/Documents/Databases/Hockey/PlayByPlay/Season_2015_2016'
REP     =   '/Users/younes_zerouali/Documents/Stradigi/Databases/Hockey/PlayByPlay/Season_2015_2016'
RSTR    =   ROSTERtable(REP, 'roster_20152016.csv')
RSTR.read()
PBP     =   PBPtable(REP, 'playbyplay_20152016.csv')
PBP.read()
PBP.detect_power_plays()

# ======
# Step1: find the Canadian games (indices start-finish)
# ======
# Get the codes for the games
P_homeTeam_ix, _    =   PBP.get_column_idx('hometeam')      # index of the column with home MTL games
P_awayTeam_ix, _    =   PBP.get_column_idx('awayteam')      # index of the column with away MTL games
P_gameCode_ix, _    =   PBP.get_column_idx('gcode')         # index of the column with game codes
P_homeGame_ix       =   PBP.filter_column(P_gameCode_ix, P_homeTeam_ix, 'MTL')
P_awayGame_ix       =   PBP.filter_column(P_gameCode_ix, P_awayTeam_ix, 'MTL')
mtlGame_code, _     =   ut_unique.main(P_homeGame_ix+P_awayGame_ix)
mtlGame_code        =   sorted(mtlGame_code)  # Codes of the Canadians games
mtlGame_home        =   [x in P_homeGame_ix for x in mtlGame_code ]
# Get the ranges for games
mtlGame_range       =   [PBP.get_column_range(P_gameCode_ix, x) for x in mtlGame_code]

# ======
# Step2: Keep the plays in 5on5
# ======
# Extract mtl games
P_type_ix, _    =   PBP.get_column_idx('powerplay')
mtlGames        =   [PBP.slice_table( mtlGame_range[x].get('start')+mtlGame_range[x].get('end'), [0, P_type_ix] ) for x in list(range(len(mtlGame_range)))]
# Detect power plays
powerplay       =   [[(x[-1]%11)!=0 for x in y] for y in mtlGames]


# ======
#  Step3: find time played by the different lineups
# ======
lineTime        =   time_lineups(mtlGames[0], PBP.header, mtlGame_home[0], powerplay[0])


# ======
#  Step4: find the fraction of exploration
# ======


# ======
#  Step5: find #points/#games won
# ======


# ======
#  Step6: correlate #points/#games won with fraction of exploration
# ======


"""
# Find position column
colPos_id   =   int( np.where([x=='pos' for x in ROSTER[0]])[0] )
colPos_dt   =   [x[colPos_id] for x in ROSTER]

# Find index column
plIx_id     =   int( np.where([x=='player.id' for x in ROSTER[0]])[0] )
plIx_dt     =   [x[plIx_id] for x in ROSTER]




# Find player lineups for the AWAY team
AWAY_plCol  =   [int(np.where([x == 'a'+str(y) for x in PbP[0]])[0]) for y in np.array(range(5))+1]
AWAY_plDt   =   [[x[y] for x in PbP[1:]] for y in AWAY_plCol]

# Find player lineups for the HOME team
HOME_plCol  =   [int(np.where([x == 'h'+str(y) for x in PbP[0]])[0]) for y in np.array(range(5))+1]
HOME_plDt   =   [[x[y] for x in PbP[1:]] for y in HOME_plCol]


### ========= QUERIES TO DB
# Find player types in AWAY team
AWAY_plPos      =   [[colPos_dt[plIx_dt.index(x)] for x in AWAY_plDt[y]] for y in np.array(range(5))]
AWAY_plPos_cnt  =   [[sum([x==y for x in AWAY_plPos[z]]) for y in plPos] for z in np.array(range(5))]

# Find player types in HOME team
HOME_plPos      =   [[colPos_dt[plIx_dt.index(x)] for x in HOME_plDt[y]] for y in np.array(range(5))]
HOME_plPos_cnt  =   [[sum([x==y for x in HOME_plPos[z]]) for y in plPos] for z in np.array(range(5))]

"""


