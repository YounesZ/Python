import pandas as pd
import numpy as np
import pickle
from sys import stdout
from os import listdir, path, scandir
from itertools import compress
from Utils.programming.ut_find_folders import *



def get_player_names(repoPbP):
    # --- Retrieve list of all players - across seasons
    # Loop on rosters and get names
    allS_p = ut_find_folders(repoPbP, True)
    plNames = [pd.read_csv(path.join(repoPbP, x, x.replace('Season', 'roster') + '.csv')) for x in allS_p]
    plNames = [x['firstlast'] for x in plNames]
    plNames = pd.concat(plNames)
    plNames.drop_duplicates(inplace=True)
    # Are strings?
    areS = [type(x) is str for x in plNames]
    plNames = plNames[areS]
    return plNames


def to_pandas(repoRaw):
    # --- List all seasons
    allS = ut_find_folders(repoRaw, True)
    # Loop on seasons
    for isea in allS:
        # --- Process summary - get header
        sumF    =   path.join(repoRaw, isea, 'summary.csv')
        # Read the file
        sumR    =   open(sumF, 'r')
        sumR    =   eval( sumR.readline().replace('null', 'None') )
        # Process the file
        sEntries=   sumR['total']
        sumRdt  =   sumR['data']
        sumKeys =   sumRdt[0].keys()
        # Turn into a pandas dataframe
        sumdt   =   [list(x.values()) for x in sumRdt]
        sumDF   =   pd.DataFrame( np.array(sumdt), columns=sumKeys )
        # Sort by teamAbbrev, playerName, gameId
        sumDF   =   sumDF.sort_values(by=['gameId', 'teamAbbrev', 'playerName'], ascending=True)
        sumDF   =   sumDF.set_index(np.arange(len(sumDF)))

        # --- Process hits - get header
        hitF    =   path.join(repoRaw, isea, 'hits.csv')
        # Read the file
        hitR    =   open(hitF, 'r')
        hitR    =   eval(hitR.readline().replace('null', 'None'))
        # Process the file
        hEntries=   hitR['total']
        hitRdt  =   hitR['data']
        hitKeys =   hitRdt[0].keys()
        # Turn into a pandas dataframe
        hitdt   =   [list(x.values()) for x in hitRdt]
        hitDF   =   pd.DataFrame(np.array(hitdt), columns=hitKeys)
        # Sort by teamAbbrev, playerName, gameId
        hitDF   =   hitDF.sort_values(by=['gameId', 'teamAbbrev', 'playerName'], ascending=True)
        hitDF   =   hitDF.set_index(np.arange(len(hitDF)))

        # --- Process penalty kills - get header
        kilF    =   path.join(repoRaw, isea, 'penalty_kills.csv')
        # Read the file
        kilR    =   open(kilF, 'r')
        kilR    =   eval(kilR.readline().replace('null', 'None'))
        # Process the file
        kEntries =  kilR['total']
        kilRdt  =   kilR['data']
        kilKeys =   kilRdt[0].keys()
        # Turn into a pandas dataframe
        kildt   =   [list(x.values()) for x in kilRdt]
        kilDF   =   pd.DataFrame(np.array(kildt), columns=kilKeys)
        # Sort by teamAbbrev, playerName, gameId
        kilDF   =   kilDF.sort_values(by=['gameId', 'teamAbbrev', 'playerName'], ascending=True)
        kilDF   =   kilDF.set_index(np.arange(len(kilDF)))

        # --- Join the two tables into a pandas series
        DF      =   pd.concat( (sumDF, hitDF, kilDF), axis=1)
        # Drop duplicates
        DF      =   DF.loc[:, ~DF.columns.duplicated()]
        # Pickle the result
        with open( path.join(repoRaw, isea, 'all_data.p'), 'wb') as f:
            pickle.dump(DF, f)


def to_player(repoRaw, repoPbP, repoPSt):
    # --- List all seasons
    allS    =   ut_find_folders(repoRaw, True)
    # --- Retrieve list of all players - across seasons
    plNames =   get_player_names(repoPbP)
    # --- Retrieve stats for each player
    count   =   0
    # Loop on players
    for pl in plNames:
        # Instantiate new player frame
        plFrame     =   pd.DataFrame()
        # Loop on seasons
        for isea in allS:
            # Load data
            datA    =   path.join(repoRaw, isea, 'all_data.p')
            with open(datA, 'rb') as f:
                datA=   pickle.load(f)
            datA['playerName']  =   datA['playerName'].apply(lambda x: x.upper())
            # Find lines w/players
            datPL   =   datA[datA.loc[:,'playerName']==pl]
            # Concatenate
            plFrame =   pd.concat( (plFrame, datPL), axis=0)
        # Save to new file
        svFile      =   path.join(repoPSt, pl.replace(' ', '_')+'.p')
        with open(svFile, 'wb') as f:
            pickle.dump(plFrame, f)
        # Print status bar
        count   +=  1
        stdout.write('\r')
        # the exact output you're looking for:
        stdout.write("Player %i/%i - %s: [%-40s] %d%%, completed" % (count, len(plNames), pl, '='*int(count/len(plNames)*40), 100*count/len(plNames)) )
        stdout.flush()


"""
# LAUNCHER
# ========
repoRaw     =   '/home/younesz/Documents/Databases/Hockey/PlayerStats/raw'
repoPbP     =   '/home/younesz/Documents/Databases/Hockey/PlayByPlay'
repoPSt     =   '/home/younesz/Documents/Databases/Hockey/PlayerStats/player'
#to_pandas(repoRaw)
to_player(repoRaw, repoPbP, repoPSt)


ss  =   '20082009'
AA  =   pd.read_csv('/home/younesz/Documents/Databases/Hockey/PlayByPlay/Season_'+ss+'/playbyplay_'+ss+'.csv')
AAa =   AA.drop_duplicates(subset='gcode')
print('All games: ',len(AAa['gcode']), '; Regular season games: ', sum(AAa['gcode']<30000))
"""

