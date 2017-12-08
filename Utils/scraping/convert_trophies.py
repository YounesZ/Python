import pandas as pd
import numpy as np
import pickle
import csv
from Utils.programming.ut_find_folders import *
from Utils.scraping.convert_names import *


def to_pandas(repoRaw):
    # --- List all seasons
    allS = ut_find_folders(repoRaw, True)
    # Loop on seasons
    for isea in allS:
        # --- ART Ross nominees
        # Load csv file
        csvF    =   path.join( repoRaw, isea, 'trophy_ross_nominees.csv' )
        cont    =   []
        with open(csvF, 'r') as f:
            csvF=   csv.reader(f, delimiter='\t')
            for row in csvF:
                cont.append(row)
        # Reshape
        cont    =   np.reshape(cont, [int(len(cont)/23), 23])
        # to PD
        clm     =   ['idx' , 'player', 'season', 'team', 'pos', 'gamesplayed', 'goals', 'assists', 'points', 'plusminus', 'penaltyminutes',\
                     'pointspergame', 'powerplaygoals', 'powerplaypoints', 'shorthandgoals', 'shorthandpoints', 'gamewinning',\
                     'overtimegoals', 'shots', 'shotspercent', 'timeOnIcePerGame', 'shiftsPerGame', 'FaceOffWinPercent']
        df_r    =   pd.DataFrame(cont, columns=clm)
        # Check names
        df_r['player']  =   df_r['player'].str.upper()
        df_r    =   convert_names(df_r, 'player')
        df_r    =   df_r.set_index('player')
        # Set places
        places          =   list( range(1,len(df_r)+1) )
        places.reverse()
        df_r['WEIGHT']   =   places

        # --- SELKE nominees
        # Load csv file
        csvF = path.join(repoRaw, isea, 'trophy_selke_nominees.csv')
        cont = []
        with open(csvF, 'r') as f:
            csvF = csv.reader(f, delimiter='\t')
            for row in csvF:
                cont.append(row)
        # to PD
        df_s    =   pd.DataFrame(cont[1:], columns=cont[0])
        # Check names
        df_s['Player']  =   df_s['Player'].str.upper()
        df_s    =   convert_names(df_s, 'Player')
        df_s    =   df_s.set_index('Player')
        # Set places
        places  =   np.array( [int(x) for x in df_s['Place'].values] )
        df_s['WEIGHT']  =   max(places) - places + 1

        # --- PICKLE IT OUT
        svname  =   path.join( repoRaw, isea, 'trophy_nominees.p')
        with open(svname, 'wb') as f:
            pickle.dump({'ross':df_r, 'selke':df_s}, f)


# LAUNCHER
# ========
# Paths
repoRaw =   '/home/younesz/Documents/Databases/Hockey/PlayerStats/raw'
to_pandas(repoRaw)