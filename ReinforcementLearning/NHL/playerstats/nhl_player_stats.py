import pickle
import pandas as pd
from os import path
from sys import stdout
from bs4 import BeautifulSoup as bs
from itertools import compress
from Utils.scraping.convert_raw import get_player_names


def pull_stats(repoPSt, repoPbP, upto, nGames):
    # Initiate empty container
    allStat     =   pd.DataFrame()
    # Get player names
    plNames     =   get_player_names(repoPbP)
    count       =   0
    # Loop on players and pull stats
    for pl in plNames:
        # Load stats file
        with open( path.join(repoPSt, pl.replace(' ', '_')+'.p'), 'rb' ) as f:
            plStat      =   pickle.load(f)
        # Sort table by date
        plStat['date']  =   [x.split('T')[0] for x in list(plStat['gameDate'])]
        plStat          =   plStat.sort_values(by='date', ascending=False)
        # Get past games
        plStat          =   plStat[plStat['date']<upto]
        # Reset indexing
        plStat          =   plStat.reset_index()
        # Select columns of interest
        dtfrm           =   plStat.loc[0:nGames, ['points', 'shots', 'faceoffWinPctg', 'plusMinus', 'hits', 'takeaways']]
        timeplayed      =   plStat['timeOnIcePerGame']
        dtfrm           =   pd.concat([dtfrm[x].div(timeplayed)*3600 for x in dtfrm], axis=1)
        dtfrm.columns   =   ['points', 'shots', 'faceoffWinPctg', 'plusMinus', 'hits', 'takeaways']
        # Add to DB
        allStat         =   pd.concat( (plNames, dtfrm.median(axis=0)), axis=0 )
        count+=1
        stdout.write('\r')
        # the exact output you're looking for:
        stdout.write("Player %i/%i - %s: [%-40s] %d%%, completed" % (count, len(plNames), pl, '=' * int(count / len(plNames) * 40), 100 * count / len(plNames)))
        stdout.flush()

        return allStat



# LAUNCHER:
# =========
repoPbP     =   '/home/younesz/Documents/Databases/Hockey/PlayByPlay'
repoPSt     =   '/home/younesz/Documents/Databases/Hockey/PlayerStats/player'
upto        =   '2012-01-01'
nGames      =   30

# Retrieve player stats up to january 1st 2012 over last 30 games
"""
PStat       =   pull_stats(repoPSt, repoPbP, upto, nGames)
"""