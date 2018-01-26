import pickle
import pandas as pd
from os import path


# This functions lists all games played in the database
def collect_gameInfos(repoPSt, repoSv, cols):
    # First look for player stats data
    PSt     =   pickle.load( open(path.join(repoPSt, 'all_players.p'), 'rb') )
    # Set columns of interest
    # Loop through players and collect data
    gI      =   pd.DataFrame()
    for ipl in PSt.keys():
        # Select appropriate columns
        subD=   PSt[ipl][cols]
        # Split date and time
        if 'gameDate' in cols:
            subD['gameTime']    =   [x.split('T')[-1] for x in subD['gameDate']]
            subD['gameDate']    =   [x.split('T')[0] for x in subD['gameDate']]
        # Append
        gI  =   pd.concat( (gI, subD), axis=0, ignore_index=True )
        # Drop duplicates
        gI  =   gI.drop_duplicates(keep='first')
    pickle.dump( gI, open(path.join(repoSv, 'gamesInfo.p'), 'wb') )



# LAUNCHER
# ========
root    =   '/home/younesz/Documents'
repoPSt =   path.join(root, 'Databases/Hockey/PlayerStats/player')
repoSv  =   path.split(path.split(repoPSt)[0])[0]

# Set columns of interest
cols    =   ['gameDate', 'gameId', 'teamAbbrev', 'opponentTeamAbbrev']
collect_gameInfos(repoPSt, repoSv, cols)
