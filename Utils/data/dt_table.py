import numpy as np
from Utils.data.dt_tools import *
from Utils.programming import ut_unique, ut_get_key_by_value

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
P_evntTeam_ix, _    =   PBP.get_column_idx('ev.team')
P_evntType_ix, _    =   PBP.get_column_idx('etype')
P_event_ix, _       =   PBP.get_column_idx('event')



# List all teams
_, allTeams     =   PBP.get_column_idx('hometeam')
allTeams, _     =   ut_unique.main(allTeams)
# Loop on all teams
ALL_timeF       =   []
ALL_shiftF      =   []
ALL_gDiff       =   []
ALL_gWon        =   []
ALL_gPoints     =   []
# Loop on all teams
for tii in allTeams:

    P_homeGame_ix, _    =   PBP.filter_column(P_gameCode_ix, P_homeTeam_ix, tii)
    P_awayGame_ix, _    =   PBP.filter_column(P_gameCode_ix, P_awayTeam_ix, tii)
    mtlGame_code, _     =   ut_unique.main(P_homeGame_ix+P_awayGame_ix)
    mtlGame_code        =   sorted(mtlGame_code)  # Codes of the Canadians games
    mtlGame_home        =   [x in P_homeGame_ix for x in mtlGame_code ]
    # Get the ranges for games
    mtlGame_range       =   [PBP.get_column_range(P_gameCode_ix, x) for x in mtlGame_code]
    mtlGame_range       =   mtlGame_range[:82]

    # ======
    # Step2: Keep the plays in 5on5
    # ======
    # Extract mtl games
    P_type_ix, _    =   PBP.get_column_idx('powerplay')
    mtlGames        =   [PBP.slice_table( mtlGame_range[x].get('start')+mtlGame_range[x].get('end'), [0, P_type_ix] ) for x in list(range(len(mtlGame_range)))]
    # Detect power plays
    powerplay       =   [[(x[-1]%11)!=0 for x in y] for y in mtlGames]

    # Debug loop
    #keeplooping = True
    #print('Looping... do your thing')
    #while keeplooping:
    #    time.sleep(0.2)

    # Loop on all games
    all_timefractions   =   [0]*len(mtlGames)
    all_shiftfractions  =   [0]*len(mtlGames)
    all_gameDiff        =   [0]*len(mtlGames)
    all_gameWon         =   [0]*len(mtlGames)
    all_gamePoints      =   [0]*len(mtlGames)
    for gii in range(len(mtlGames)):

        # ======
        #  Step3: find time and number of shifts played by the different lineups
        # ======
        lineTime        =   time_lineups(mtlGames[gii], PBP.header, mtlGame_home[gii], powerplay[gii])
        lineShifts      =   shift_lineups(mtlGames[gii], PBP.header, mtlGame_home[gii], powerplay[gii])

        # ======
        #  Step4: find the fraction of exploration
        # ======
        # By time
        greedyLine_time =   sorted(lineTime.values(), reverse=True)[:4]
        greedyLine_id   =   ut_get_key_by_value.main(lineTime, greedyLine_time)
        all_timefractions[gii]  =   1 - sum([lineTime[x] for x in greedyLine_id]) / sum(list(lineTime.values()))
        # By shifts
        greedyLine_shifts       =   sorted(lineShifts.values(), reverse=True)[:4]
        all_shiftfractions[gii] =   1 - sum(greedyLine_shifts) / sum(list(lineShifts.values()))

        # ======
        #  Step5: find #points/#games won
        # ======
        # Find overtime
        _, idxPauses=   PBP.filter_column(P_event_ix, P_evntType_ix, 'PEND', mtlGames[gii])
        # Get goal: regular time
        gameGoals,_ =   PBP.filter_column(P_evntTeam_ix, P_evntType_ix, 'GOAL', mtlGames[gii][:int(idxPauses[min(2,len(idxPauses)-1)])])
        ggoals_for  =   sum([x==tii for x in gameGoals])
        ggoals_against=  sum([x!=tii for x in gameGoals])
        # Get goal: over time
        otGoals,_   =   PBP.filter_column(P_evntTeam_ix, P_evntType_ix, 'GOAL', mtlGames[gii][int(idxPauses[min(2,len(idxPauses)-1)]):])
        ogoals_for  =   sum([x==tii for x in otGoals])
        ogoals_against=  sum([x!=tii for x in otGoals])
        # Get goal differential
        all_gameDiff[gii]   =   ggoals_for - ggoals_against + int(np.sign(ogoals_for-ogoals_against))
        # Get game result
        all_gameWon[gii]    =   int( np.sign(all_gameDiff[gii]) )
        # Number of points taken
        all_gamePoints[gii] =   int((ggoals_for-ggoals_against)>0) + int((ggoals_for-ggoals_against)>=0) + int((ogoals_for-ogoals_against)>0)

    # Append data
    ALL_timeF.append(all_timefractions)
    ALL_shiftF.append(all_shiftfractions)
    ALL_gDiff.append(all_gameDiff)
    ALL_gWon.append(all_gameWon)
    ALL_gPoints.append(all_gamePoints)

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


