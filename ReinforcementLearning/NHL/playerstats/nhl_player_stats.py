import csv
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path
from sys import stdout
from copy import deepcopy
from Utils.scraping.convert_raw import get_player_names
from Utils.programming.ut_find_folders import *

def pull_stats(repoPSt, repoPbP, asof='2001-09-01', upto='2016-07-01', nGames=82):
    # Initiate empty container
    allStat     =   pd.DataFrame()
    allPos      =   []
    # Get player names
    plNames     =   get_player_names(repoPbP)
    count       =   0
    # Prep
    tobesum     =   ['gamesPlayed', 'goals', 'assists', 'points', 'plusMinus', 'penaltyMinutes', 'ppGoals', 'ppPoints', \
                     'shGoals', 'shPoints', 'gameWinningGoals', 'otGoals', 'shots', 'shGoals', 'shAssists', 'shPoints', \
                     'shShots', 'shHits', 'shBlockedShots', 'shMissedShots', 'shGiveaways', 'shTakeaways', 'shFaceoffsWon', \
                     'shFaceoffsLost', 'hits', 'blockedShots', 'missedShots', 'giveaways', 'takeaways', 'faceoffs', \
                     'faceoffsWon', 'faceoffsLost']
    tobeavg     =   ['shootingPctg', 'timeOnIcePerGame', 'shiftsPerGame', 'faceoffWinPctg', 'shTimeOnIce','hitsPerGame', \
                     'blockedShotsPerGame', 'missedShotsPerGame', 'faceoffWinPctg', 'shotsPerGame']
    #tobenorm    =   ['goals', 'penaltyMinutes','gameWinningGoals', 'ppGoals', 'ppPoints', 'assists','shGoals', 'shots', 'shPoints', 'points', 'plusMinus', 'faceoffsLost', 'blockedShots', 'missedShots', 'hits', 'takeaways', 'giveaways', 'faceoffsWon', 'faceoffs', 'shFaceoffsLost', 'shBlockedShots', 'shGiveaways', 'shMissedShots', 'shTakeaways', 'shFaceoffsWon', 'shShots', 'shAssists', 'shHits', 'shTimeOnIce']
    #tobeavg     =   ['goals', 'penaltyMinutes','gameWinningGoals', 'faceoffWinPctg', 'ppGoals', 'ppPoints', 'assists', 'shGoals', 'shootingPctg', 'shots', 'shPoints', 'points', 'plusMinus', 'otGoals', 'timeOnIcePerGame', 'shiftsPerGame', 'gamesPlayed', 'missedShotsPerGame', 'goalsPerGame', 'faceoffsLost', 'blockedShots', 'shotsPerGame', 'missedShots', 'hitsPerGame', 'hits', 'takeaways', 'blockedShotsPerGame', 'giveaways', 'faceoffsWon', 'faceoffs', 'shFaceoffsLost', 'shBlockedShots', 'shGiveaways', 'shMissedShots', 'shTakeaways', 'shFaceoffsWon', 'shShots', 'shAssists', 'shHits', 'shTimeOnIce']
    #nottobenorm =   list( set(list(tobeavg)).difference(tobenorm) )
    # Loop on players and pull stats
    for pl in plNames:
        # Load stats file
        with open( path.join(repoPSt, pl.replace(' ', '_')+'.p'), 'rb' ) as f:
            plStat      =   pickle.load(f)
        # Sort table by date
        plStat['date']  =   [x.split('T')[0] for x in list(plStat['gameDate'])]
        plStat          =   plStat.sort_values(by='date', ascending=False)
        # Get past games
        if len(plStat['date'])>0:
            plStat      =   plStat[plStat['date']>=asof]
            plStat      =   plStat[plStat['date']<=upto]
            # Reset indexing
            plStat          =   plStat.reset_index()
            nottobeavg      =   list( set(list(plStat.columns)).difference(tobeavg) )
            nottobeavg      =   list( set(list(plStat.columns)).difference(tobesum).difference(tobeavg) )
            # Select columns of interest
            plStat          =   plStat.loc[0:nGames, :]
            timeplayed      =   deepcopy( plStat.loc[0:nGames, 'timeOnIcePerGame'] )
            # Remove games where the TOI was 0
            plStat          =   plStat[timeplayed>0]
            timeplayed      =   timeplayed[timeplayed>0]
            if len(plStat)>0:
                # Init new dataframe
                columns     =   plStat.columns
                plStat      =   plStat.reset_index()
                newDF       =   pd.DataFrame(columns=columns)
                newDF.loc[0, nottobeavg] = plStat.loc[0, nottobeavg]
                """
                # Normalize stats by time played
                dtfrm       = pd.concat([plStat[x].div(timeplayed) * 3600 for x in tobenorm], axis=1)
                dtfrm.columns = tobenorm
                dtfrm       = pd.concat([dtfrm, pd.concat([plStat[x] for x in nottobenorm], axis=1)], axis=1)
                """
                # Average columns
                newDF.loc[0, tobeavg]   =   plStat[tobeavg].mean(axis=0)
                newDF.loc[0, tobesum]   =   plStat[tobesum].sum(axis=0)
                newDF['player'] =   pl
                dtpos           =   plStat.loc[0,'playerPositionCode']
                # Add to DB
                allStat         =   pd.concat( (allStat, newDF), axis=0, ignore_index=True )
                allPos.append( dtpos )
        count+=1
        if count % 100 == 0:
            stdout.write('\r')
            # the exact output you're looking for:
            stdout.write("Player %i/%i - %s: [%-40s] %d%%, completed" % (count, len(plNames), pl, '=' * int(count / len(plNames) * 40), 100 * count / len(plNames)))
            stdout.flush()
    allStat     =   allStat.set_index('player')
    allPos      =   pd.DataFrame(allPos, columns=['playerPositionCode'])
    return allStat, allPos, tobeavg, tobesum


def to_quartiles(stats):
    # Init new data frame
    quarts  =   pd.DataFrame()
    # Loop on columns
    for col in stats.columns:
        # Find boundaries
        dt  =   stats[col]
        dt  =   np.unique( np.sort(dt) )
        bnd =   [min(dt)-1] + list(dt[np.int16(len(dt) * (np.array(range(3))+1) / 4 -1)])
        # Find quartile
        qrt =   np.zeros( np.shape(stats[col]) )
        for ib in bnd:
            qrt     +=  stats[col]>ib
        # Add to dataframe
        quarts[col]     =   qrt
    return quarts


def as_matrix(defense, offense):
    # Convert to matrix
    yax = list(np.unique(defense))
    xax = list(np.unique(offense))
    mtx = np.zeros([len(yax), len(xax)])
    for ix, iy in zip(offense, defense):
        mtx[yax.index(iy), xax.index(ix)] += 1
    plt.figure();
    plt.imshow(mtx)
    plt.gca().set_xlabel('Offensive ranking (quartile)')
    plt.gca().set_ylabel('Defensive ranking (quartile)')
    plt.gca().set_xticklabels(xax)
    plt.gca().set_yticklabels([0] + yax)
    plt.gca().set_xticklabels([0] + xax)
    plt.gca().invert_yaxis()


def to_classes(PL, stOff, stDef, qthresh=3.5, by='highest_value'):
    # Offensive vs Defensive ranking
    offR    =   PL[stOff].mean(axis=1)
    defR    =   PL[stDef].mean(axis=1)
    #as_matrix(defR, offR)
    if by=='highest_value':
        # Classify players: HIGHEST VALUE
        isOFF   =   (offR > defR).values
        isGOOD  =   np.maximum(offR.values, defR.values) < qthresh
        CLASS   =   np.array([2 * int(x) - 1 for x in isOFF])
        CLASS[isGOOD]   =   0
    elif by=='defensive_first':
        # Classify players: defensive skills first
        CLASS   =   np.zeros(len(offR))
        CLASS[np.array(offR.values) >= qthresh]     =   1
        CLASS[np.array(defR.values) >= qthresh]     =   -1
    return CLASS


def validate_classes(PLclass, PLnames, repoRaw, season):
    # Load reference data
    refD    =   path.join( repoRaw, season, 'trophy_nominees.p')
    with open(refD, 'rb') as f:
        refD=   pickle.load(f)
    # Prep checks
    offNames =  PLnames[PLclass == 1]
    defNames =  PLnames[PLclass == -1]
    ntrNames =  PLnames[PLclass == 0]
    # --- Check OFFENSIVE skills
    ross_nominees   =   [x.upper() for x in list( refD['ross'].index )]
    confusion_off   =   [sum([x in y for x in ross_nominees])/len(ross_nominees) for y in [offNames,  ntrNames, defNames]]
    # --- Check DEFENSIVE skills
    selke_nominees  =   [x.upper() for x in list(refD['selke'].index)]
    confusion_def   =   [sum([x in y for x in selke_nominees])/len(selke_nominees) for y in [offNames,  ntrNames, defNames]]
    # Display as matrix
    FIG     =   plt.figure()
    Ax1     =   FIG.add_subplot(211)
    Ax1.imshow(np.reshape(confusion_off, [1, len(confusion_off)]))
    Ax1.set_title('Confusion matrices')
    Ax1.set_ylabel('Art Ross nominees')
    Ax1.set_yticklabels('')
    Ax2     =   FIG.add_subplot(212)
    Ax2.imshow(np.reshape(confusion_def, [1, len(confusion_def)]))
    Ax2.set_ylabel('Frank J. Selke nominees')
    Ax2.set_xticklabels(['', 'Offensive', '', 'Neutral', '', 'Defensive'])
    Ax2.set_yticklabels('')


def do_manual_classification(repoPSt, repoPbP, upto, nGames):
    # Retrieve player stats up to january 1st 2012 over last 30 games
    PStat, PPos, numericCols, normalizedCols = pull_stats(repoPSt, repoPbP, upto, nGames)
    # Convert stats to quartiles
    PQuart      =   to_quartiles(PStat)
    # Select offensive players
    PQuart_off  =   PQuart[(PPos!='D').values]
    # Classify players
    critOff     =   ['faceoffWinPctg', 'points', 'shots']     #
    critDef     =   ['hits', 'blockedShots', 'takeaways' , 'plusMinus', 'shTimeOnIce']    # ,
    PLclass     =   to_classes(PQuart_off, critOff, critDef, qthresh=3, by='highest_value')
    return PLclass, PQuart_off.index


def do_ANN_classification(repoPSt, repoPbP):
    # List non-lockout seasons
    allS_p  =   ut_find_folders(repoPbP, True)
    # Prep dataset
    """dtCols  =   ['ppPoints', 'plusMinus', 'otGoals', 'points', 'shootingPctg', 'goals', 'faceoffWinPctg', 'penaltyMinutes',\
                'shPoints', 'shots', 'gamesPlayed', 'ppGoals', 'timeOnIcePerGame', 'gameWinningGoals', 'assists',\
                'shGoals', 'shiftsPerGame', 'faceoffsWon', 'hits', 'missedShots', 'hitsPerGame', 'shotsPerGame', 'blockedShots',\
                'giveaways', 'blockedShotsPerGame', 'goalsPerGame', 'missedShotsPerGame', 'faceoffsLost', 'takeaways',\
                'faceoffs', 'shShots', 'shFaceoffsLost', 'shHits', 'shGiveaways', 'shTimeOnIce', 'shAssists', 'shFaceoffsWon',\
                'shBlockedShots', 'shTakeaways', 'shMissedShots']"""
    X       =   pd.DataFrame()
    Y       =   np.zeros([0,2])
    # Loop on seasons and collect data
    for isea in allS_p:
        # Get end time stamp
        sea_name    =   isea.replace('Season_', '')
        sea_strt    =   sea_name[:4] + '-09-01'
        sea_end     =   sea_name[-4:] + '-07-01'
        # Pull stats
        sea_stats, sea_pos, sea_numeric, sea_normal     =   pull_stats(repoPSt, repoPbP, sea_strt, sea_end)
        dtCols      =   list( set(sea_numeric).union(sea_normal) )
        # Pull Selke and Ross nominees for that season
        with open( path.join(repoPSt.replace('player', 'raw'), sea_name, 'trophy_nominees.p'), 'rb') as f:
            trophies=   pickle.load(f)
        # Process Selke
        df_s        =   trophies['selke']
        dt_selke    =   sea_stats.loc[df_s.index.values, dtCols]
        # Process Ross
        df_r        =   trophies['ross']
        dt_ross     =   sea_stats.loc[df_r.index.values, dtCols]
        # Append to dataset
        X           =   pd.concat( (X, dt_ross, dt_selke), axis=0)
        tempy1      =   np.concatenate( ( np.reshape(df_r['WEIGHT'].values, [len(df_r),1]), np.zeros([len(df_r),1])), axis=1 )
        tempy2      =   np.concatenate( (np.zeros([len(df_s), 1]), np.reshape(df_s['WEIGHT'].values, [len(df_s), 1])), axis=1)
        Y           =   np.vstack( [Y, tempy1] )
        Y           =   np.vstack([Y, tempy2])
    return X, Y


# LAUNCHER:
# =========
repoPbP     =   '/home/younesz/Documents/Databases/Hockey/PlayByPlay'
repoPSt     =   '/home/younesz/Documents/Databases/Hockey/PlayerStats/player'
repoRaw     =   '/home/younesz/Documents/Databases/Hockey/PlayerStats/raw'

# Make classification - ANN
X, Y    =   do_ANN_classification(repoPSt, repoPbP)

# --- BUILD ANN
# Input
inpSize     =   np.shape(X)[1]
annLay      =   [10,5]
# Architecture
annX        =   tf.placeholder(tf.float32, [None, inpSize, 1], name='Input to the network - player features')
annY_       =   tf.placeholder(tf.float32, [None, 2], name='Ground truth')
annW1       =   tf.Variable( tf.truncated_normal([inpSize, annLay[0]], stddev=0.1) )
annB1       =   tf.Variable( tf.ones([1, annLay[0]])/10 )
Y1          =   tf.matmul( annW1, X) + annB1
annW2       =   tf.Variable( tf.truncated_normal([annLay[0], annLay[1]], stddev=0.1) )
annB2       =   tf.Variable( tf.ones([1, annLay[1]])/10 )
Y2          =   tf.matmul( annW2, Y1) + annB2
annW3       =   tf.Variable( tf.truncated_normal([annLay[1], 2], stddev=0.1) )
annB3       =   tf.Variable( tf.ones([1, annLay[1]])/10 )
annY        =   tf.matmul( annW3, Y2) + annB3
# Init variables
init        =   tf.initialize_all_variables()
# Optimization
loss        =   tf.losses.mean_squared_error(annY_, annY)
optimizer   =   tf.train.AdamOptimizer(0.01)
# Compute accuracy
accuracy    =   tf.reduce_mean( annY-annY_ )

"""
# Make classification - manual
upto    =   '2013-07-01'
nGames  =   80
PLclass, PLnames    =   do_manual_classification(repoPSt, repoPbP, upto, nGames)

# Sanity check
season  =   '20122013'
validate_classes(PLclass, PLnames, repoRaw, season)
"""

"""
PQuart_off.index[CLASS==0]
"""