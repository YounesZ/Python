import csv
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path
from sys import stdout
from copy import deepcopy
from sklearn import preprocessing
from itertools import compress
from sklearn.decomposition import PCA
from Utils.maths.ut_cumsum_thresh import *
from Utils.clustering.ut_make_constraints import *
from Utils.clustering.ut_center_of_mass import *
from Utils.programming.ut_find_folders import *
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.builder import SavedModelBuilder
from Utils.programming.ut_difference import *
from Utils.programming.ut_sanitize_matrix import ut_sanitize_matrix
from Utils.scraping.convert_raw import get_player_names
from Utils.scraping.convert_trophies import to_pandas_selke, to_pandas_ross
from Clustering.copkmeans.cop_kmeans import cop_kmeans



class ANN_classifier():

    def __init__(self, nNodes=[10,20,2]):
        # Launch the builder
        nodes       =   deepcopy(nNodes)
        nInputs     =   nodes.pop(0)
        nOutputs    =   nodes.pop(-1)
        nHidden     =   len(nodes)
        self.ann_build_network(nInputs, nOutputs, nodes)

    def ann_build_network(self, nInputs, noutputs, nNodes):
        # Architecture - 1 layer
        self.annX   =   tf.placeholder(tf.float32, [None, nInputs], name='Input_to_the_network-player_features')
        self.annY_  =   tf.placeholder(tf.float32, [None, 2], name='Ground_truth')
        annW1       =   tf.Variable(tf.truncated_normal([nInputs, nNodes[0]], stddev=0.1), name='weights_inp_hid')
        annB1       =   tf.Variable(tf.ones([1, nNodes[0]]) / 10, name='bias_inp_hid')
        Y1          =   tf.add( tf.nn.relu(tf.matmul(self.annX, annW1)), annB1, name='hid_output')
        annW2       =   tf.Variable(tf.truncated_normal([nNodes[0], noutputs], stddev=0.1), name='weights_hid_out')
        annB2       =   tf.Variable(tf.ones([1, noutputs]) / 10, name='bias_hid_out')
        self.annY   =   tf.add( tf.matmul(Y1, annW2), annB2, name='prediction' )
        # Init variables
        init        =   tf.global_variables_initializer()
        # Optimization
        self.loss       =   tf.reduce_mean(tf.squared_difference(self.annY_, self.annY))
        self.train_step =   tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
        # Compute accuracy
        is_correct      =   tf.equal(tf.argmax(self.annY, axis=1), tf.argmax(self.annY_, axis=1))
        self.accuracy   =   tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def ann_train_network(self, nIter, annI, annT, svname=None):
        self.trLoss, self.tsLoss, self.trAcc, self.tsAcc, nIter = [], [], [], [], 50
        # Initialize the model saver
        #builder         =   SavedModelBuilder(svname)
        saver           =   tf.train.Saver()
        for iIt in range(nIter):
            # --- TRAIN ANN
            # Init instance
            init        =   tf.global_variables_initializer()
            self.sess   =   tf.Session()
            self.sess.run(init)
            # Split train/test data
            train_X, test_X, train_Y, test_Y = train_test_split(annI, annT, test_size=0.25)
            # Loop on data splits
            fcnLoss =   []
            # Make exponential batch size increase
            batchSize, minSize, maxSize, nSteps =   [], 5, 20, 0
            while np.sum(batchSize) + maxSize < train_X.shape[0]:
                nSteps      +=  1
                batchSize   =   np.floor( ((np.exp(range(nSteps)) - 1) / (np.exp(nSteps) - 1)) ** .05 * (maxSize - minSize)) + minSize
            batchSize   =   np.append(batchSize, train_X.shape[0] - np.sum(batchSize)).astype(int)
            trL, tsL, trA, tsA  =   [], [], [], []
            for ib in batchSize:
                # Slice input and target
                trInput, train_X    =   train_X[:ib, :], train_X[ib:, :]
                trTarget, train_Y   =   train_Y[:ib, :], train_Y[ib:, :]
                # Pass them through
                dictDT  =   {self.annX: trInput, self.annY_: trTarget}
                self.sess.run(self.train_step, feed_dict=dictDT)
                trL.append(self.sess.run(self.loss, feed_dict=dictDT))
                trA.append(self.sess.run(self.accuracy, feed_dict=dictDT))
                # Assess accuracy
                dictTS  =   {self.annX: test_X, self.annY_: test_Y}
                tsL.append(self.sess.run(self.loss, feed_dict=dictTS))
                tsA.append(self.sess.run(self.accuracy, feed_dict=dictTS))
            # plt.figure(); plt.plot(fcnLoss)
            self.trLoss.append(trL)
            self.tsLoss.append(tsL)
            self.trAcc.append(trA)
            self.tsAcc.append(tsA)
            self.batchSize  =   batchSize
        # Save session
        if not svname is None:
            """
            builder.add_meta_graph_and_variables(self.sess, [tag_constants.SERVING])
            builder.save()
            """
            saver.save(self.sess, path.join(repoModel, 'MODEL_perceptron_1layer_10units_relu'))
            pickle.dump({'trLoss':self.trLoss, 'tsLoss':self.tsLoss, 'trAcc':self.trAcc, 'tsAcc':self.tsAcc, 'batchSize':self.batchSize}, \
                        open(path.join(repoModel, 'addedVariables.p'), 'wb') )

    def ann_display_accuracy(self):
        # Make figure
        Fig     =   plt.figure()
        # Axes1: loss
        Ax1     =   Fig.add_subplot(121)
        Ax1.fill_between(np.cumsum(self.batchSize), np.mean(self.trLoss, axis=0) - np.std(self.trLoss, axis=0), np.mean(self.trLoss, axis=0) + np.std(self.trLoss, axis=0), facecolors='b', interpolate=True, alpha=0.4)
        Ax1.fill_between(np.cumsum(self.batchSize), np.mean(self.tsLoss, axis=0) - np.std(self.tsLoss, axis=0), np.mean(self.tsLoss, axis=0) + np.std(self.tsLoss, axis=0), facecolors='r', interpolate=True, alpha=0.4)
        Ax1.plot(np.cumsum(self.batchSize), np.mean(self.trLoss, axis=0), 'b')
        Ax1.plot(np.cumsum(self.batchSize), np.mean(self.tsLoss, axis=0), 'r')
        Ax1.set_xlabel('Number of training examples')
        Ax1.set_ylabel('Quadratic error')
        Ax1.set_xlim([min(self.batchSize), np.sum(self.batchSize)])
        Ax1.set_ylim([0.04, 0.16])
        # Axes2: accuracy
        Ax2     =   Fig.add_subplot(122)
        Ax2.fill_between(np.cumsum(self.batchSize), np.mean(self.trAcc, axis=0) - np.std(self.trAcc, axis=0), np.mean(self.trAcc, axis=0) + np.std(self.trAcc, axis=0), facecolors='b', interpolate=True, alpha=0.4)
        Ax2.fill_between(np.cumsum(self.batchSize), np.mean(self.tsAcc, axis=0) - np.std(self.tsAcc, axis=0), np.mean(self.tsAcc, axis=0) + np.std(self.tsAcc, axis=0), facecolors='r', interpolate=True, alpha=0.4)
        Ax2.plot(np.cumsum(self.batchSize), np.mean(self.trAcc, axis=0), 'b')
        Ax2.plot(np.cumsum(self.batchSize), np.mean(self.tsAcc, axis=0), 'r')
        Ax2.set_xlabel('Number of training examples')
        Ax2.set_ylabel('Classification accuracy')
        Ax2.set_xlim([min(self.batchSize), np.sum(self.batchSize)])
        Ax2.set_ylim([0.4, 1])

    def ann_reload_network(self, repoModel):
        # Reload the graph and variables
        self.sess   =   tf.Session()
        saver       =   tf.train.import_meta_graph( path.join(repoModel, path.basename(repoModel)+'.meta') )
        saver.restore(self.sess, tf.train.latest_checkpoint(path.join(repoModel, './')))
        # Link TF variables to the classifier class
        graph       =   self.sess.graph
        self.annX   =   graph.get_tensor_by_name('Input_to_the_network-player_features:0')
        """self.annY_  =   graph.get_tensor_by_name('Ground_truth:0')
        self.annW1  =   graph.get_tensor_by_name('weights_inp_hid:0')
        self.annB1  =   graph.get_tensor_by_name('bias_inp_hid:0')
        self.Y1     =   graph.get_operation_by_name('hid_output')
        self.annW2  =   graph.get_tensor_by_name('weights_hid_out:0')
        self.annB2  =   graph.get_tensor_by_name('bias_hid_out:0')"""
        self.annY   =   graph.get_tensor_by_name('prediction:0')
        # Restore additional variables
        VAR         =   pickle.load( open(path.join(repoModel, 'addedVariables.p'), 'rb') )
        self.trLoss =   VAR['trLoss']
        self.tsLoss =   VAR['tsLoss']
        self.trAcc  =   VAR['trAcc']
        self.tsAcc  =   VAR['tsAcc']
        self.batchSize = VAR['batchSize']


def pull_stats(repoPSt, repoPbP, asof='2001-09-01', upto='2016-07-01', uptocode=None, nGames=82, plNames=None):

    # Get player names
    depickle        =   True
    if plNames is None:
        plNames     =   get_player_names(repoPbP)
        # De-pickle all players
        depickle    =   False
        with open(path.join(repoPSt, 'all_players.p'), 'rb') as f:
            all_pl  =   pickle.load(f)
    count   =   0
    # Prep
    """
    tobesum     =   ['gamesPlayed', 'goals', 'assists', 'points', 'plusMinus', 'penaltyMinutes', 'ppGoals', 'ppPoints', \
                     'shGoals', 'shPoints', 'gameWinningGoals', 'otGoals', 'shots', 'shGoals', 'shAssists', 'shPoints', \
                     'shShots', 'shHits', 'shBlockedShots', 'shMissedShots', 'shGiveaways', 'shTakeaways', 'shFaceoffsWon', \
                     'shFaceoffsLost', 'hits', 'blockedShots', 'missedShots', 'giveaways', 'takeaways', 'faceoffs', \
                     'faceoffsWon', 'faceoffsLost']
    """
    tobeavg     =   ['shootingPctg', 'shiftsPerGame', 'faceoffWinPctg', 'hitsPerGame', 'blockedShotsPerGame', 'missedShotsPerGame', 'shotsPerGame']
    tobenorm_rs =   ['penaltyMinutes', 'assists', 'goals', 'gameWinningGoals', 'points', 'shots', 'otGoals', 'missedShots', 'hits', 'takeaways', 'faceoffsWon', 'blockedShots', 'faceoffsLost', 'giveaways', 'faceoffs']
    tobenorm_pp =   ['ppPoints', 'ppGoals', 'ppShots', 'ppGiveaways', 'ppTakeaways', 'ppHits', 'ppFaceoffsWon', 'ppMissedShots', 'ppFaceoffsLost', 'ppTimeOnIce', 'ppAssists']
    tobenorm_pk =   ['shGoals', 'shPoints', 'shMissedShots', 'shHits', 'shFaceoffsWon', 'shShots', 'shGiveaways', 'shFaceoffsLost', 'shBlockedShots', 'shTakeaways', 'shAssists']
    columns     =   tobeavg + tobenorm_rs + tobenorm_pp + tobenorm_pk

    # Initiate empty container
    allStat     =   pd.DataFrame( columns = columns+['player', 'position', 'gmPl'] )
    allPos      =   []
    allGP       =   []
    #tobeavg     =   ['goals', 'penaltyMinutes','gameWinningGoals', 'faceoffWinPctg', 'ppGoals', 'ppPoints', 'assists', 'shGoals', 'shootingPctg', 'shots', 'shPoints', 'points', 'plusMinus', 'otGoals', 'timeOnIcePerGame', 'shiftsPerGame', 'gamesPlayed', 'missedShotsPerGame', 'goalsPerGame', 'faceoffsLost', 'blockedShots', 'shotsPerGame', 'missedShots', 'hitsPerGame', 'hits', 'takeaways', 'blockedShotsPerGame', 'giveaways', 'faceoffsWon', 'faceoffs', 'shFaceoffsLost', 'shBlockedShots', 'shGiveaways', 'shMissedShots', 'shTakeaways', 'shFaceoffsWon', 'shShots', 'shAssists', 'shHits']
    #nottobenorm =   list( set(list(tobeavg)).difference(tobenorm) )

    for pl in plNames:
        # Load stats file
        if depickle:
            plStat      =   pickle.load( open( path.join(repoPSt, pl.replace(' ', '_')+'.p'), 'rb') )
        else:
            plStat      =   all_pl[pl]
        # Sort table by date
        plStat['date']  =   [x.split('T')[0] for x in list(plStat['gameDate'])]
        plStat          =   plStat.sort_values(by='date', ascending=False)
        # Get past games
        newDF           =   pd.DataFrame(np.zeros([1, len(columns)]), columns=columns)
        newDF['player'] =   pl
        newDF['position']=  'U'
        newDF['gmPl']   =   0
        if len(plStat['date'])>0:
            plStat      =   plStat[plStat['date']>=asof]
            plStat      =   plStat[plStat['date']<=upto]
            if not uptocode is None:
                plStat  =   plStat[plStat['gameId']<uptocode]
            # Reset indexing
            plStat          =   plStat.reset_index()
            #nottobeavg      =   list( set(list(plStat.columns)).difference(tobeavg) )
            #nottobeavg      =   list( set(list(plStat.columns)).difference(tobesum).difference(tobeavg) )
            # Select columns of interest
            plStat          =   plStat.loc[0:nGames, :]
            timeplayed_rs   =   deepcopy( plStat.loc[0:nGames, 'timeOnIcePerGame'] )
            #timeplayed_pp   =   deepcopy(plStat.loc[0:nGames, 'ppTimeOnIce'])
            #timeplayed_pk   =   deepcopy(plStat.loc[0:nGames, 'shTimeOnIce'])
            # Remove games where the TOI was 0
            plStat          =   plStat[timeplayed_rs>0]
            #timeplayed_pp   =   timeplayed_pp[timeplayed_rs>0]
            #timeplayed_pk   =   timeplayed_pk[timeplayed_rs>0]
            #timeplayed_rs   =   timeplayed_rs[timeplayed_rs>0]
            if len(plStat)>0:
                # Init new dataframe
                plStat      =   plStat.reset_index()
                """
                newDF       =   pd.DataFrame( np.zeros([1, len(columns)]), columns=columns)
                # Normalize regular stats by time played
                for iC in tobenorm_rs:
                    newDF[iC]   =   ( plStat[iC].div(timeplayed_rs) * 3600 ).mean(axis=0)                    
                # Normalize powerplay stats by time played
                for iC in tobenorm_pp:
                    newDF[iC]   =   ( plStat[iC][(timeplayed_pp > 0).values].div(timeplayed_pp[timeplayed_pp > 0]) * 3600).sum() / len(plStat)
                    if newDF[iC].isnull().any(): newDF[iC] = 0
                # Normalize penalty kill stats by time played
                for iC in tobenorm_pk:
                    newDF[iC]   =   ( plStat[iC][(timeplayed_pk > 0).values].div(timeplayed_pk[timeplayed_pk>0]) * 3600 ).sum()/len(plStat)                    
                    if newDF[iC].isnull().any(): newDF[iC]    =   0
                # Normalize by number of games played
                for iC in columns:
                    newDF[iC] = plStat[iC].sum(axis=0) / len(plStat)
                # Average columns
                newDF[tobeavg]  =   plStat[tobeavg].mean(axis=0).values
                """
                tbavg       =   plStat[tobeavg].sum(axis=0).values / len(plStat)
                tbnrm_rs    =   plStat[tobenorm_rs].sum(axis=0).values / len(plStat)
                tbnrm_pp    =   plStat[tobenorm_pp].sum(axis=0).values / len(plStat)
                tbnrm_pk    =   plStat[tobenorm_pk].sum(axis=0).values / len(plStat)
                allV        =   np.concatenate( (tbavg, tbnrm_rs, tbnrm_pp, tbnrm_pk) )
                newDF       =   pd.DataFrame( np.reshape(allV, [1, len(columns)]), columns=columns)
                newDF['player']     =   pl
                newDF['position']   =  plStat.loc[0,'playerPositionCode']
                newDF['gmPl']       =   np.sum(plStat['gamesPlayed'])
        # Add to DB
        allStat =   pd.concat( (allStat, newDF), axis=0, ignore_index=True )

        count+=1
        if count % 500 == 0:
            stdout.write('\r')
            # the exact output you're looking for:
            stdout.write("Player %i/%i - %s: [%-40s] %d%%, completed" % (count, len(plNames), pl, '=' * int(count / len(plNames) * 40), 100 * count / len(plNames)))
            stdout.flush()
    allStat     =   allStat.set_index('player')
    return allStat, columns


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
    PStat, PPos, GmPl = pull_stats(repoPSt, repoPbP, upto, nGames)
    # Convert stats to quartiles
    PQuart      =   to_quartiles(PStat)
    # Select offensive players
    PQuart_off  =   PQuart[(PPos!='D').values]
    # Classify players
    critOff     =   ['faceoffWinPctg', 'points', 'shots']     #
    critDef     =   ['hits', 'blockedShots', 'takeaways' , 'plusMinus', 'shTimeOnIce']    # ,
    PLclass     =   to_classes(PQuart_off, critOff, critDef, qthresh=3, by='highest_value')
    return PLclass, PQuart_off.index


def get_training_data(season):
    X_train     =   pd.DataFrame()
    Y_train     =   pd.DataFrame()
    X_all       =   pd.DataFrame()
    POS_all     =   pd.DataFrame()
    # Loop on seasons and collect data
    for isea in season:
        # Get end time stamp
        sea_name=   isea.replace('Season_', '')
        sea_strt=   sea_name[:4] + '-09-01'
        sea_end =   sea_name[-4:] + '-07-01'
        # Pull stats
        sea_stats, dtCols   =   pull_stats(repoPSt, repoPbP, asof=sea_strt, upto=sea_end)
        # Pull Selke and Ross nominees for that season
        with open(path.join(repoPSt.replace('player', 'raw'), sea_name, 'trophy_nominees.p'), 'rb') as f:
            trophies = pickle.load(f)
        # Process Selke
        df_s    =   trophies['selke']
        # Process Ross
        df_r    =   trophies['ross']
        # Make the voting dataframes
        tempy1  =   np.concatenate((np.reshape(df_r['WEIGHT'].values, [len(df_r), 1]), np.zeros([len(df_r), 1])), axis=1)
        tempy2  =   np.concatenate((np.zeros([len(df_s), 1]), np.reshape(df_s['WEIGHT'].values, [len(df_s), 1])), axis=1)
        tempY   =   pd.concat([pd.DataFrame(tempy1/np.max(tempy1), index=df_r.index), pd.DataFrame(tempy2/np.max(tempy2), index=df_s.index)])
        tempY   =   tempY.groupby(tempY.index).agg({0:sum, 1:sum})
        tempX   =   sea_stats.loc[tempY.index, dtCols]
        # Append to dataset
        X_train =   pd.concat((X_train, tempX), axis=0)
        Y_train =   pd.concat([Y_train, tempY], axis=0)
        X_all   =   pd.concat([X_all, sea_stats[dtCols]])
        POS_all =   pd.concat([POS_all, sea_stats['position']])
    return X_train, Y_train, X_all, POS_all, dtCols


def do_normalize_data(data, mu=None, sigma=None, normalizer=None):
    # Custom normalization
    wtd     =   ['max', 'minmax', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max']
    donorm  =   False
    # Remove NaNs
    data    =   data.values
    if normalizer is None:
        normalizer  =   [[]] * len(wtd)
        donorm      =   True
    for ii in range( data.shape[1] ):
        todo    =   wtd[ii]
        vec     =   data[:, ii]
        if todo=='max':
            if donorm: normalizer[ii] = np.max(vec)
            data[:,ii]  =   vec / normalizer[ii]
        elif todo=='minmax':
            if donorm: normalizer[ii] = [np.min(vec), np.max(vec)]
            data[:,ii]  =   (vec - normalizer[ii][0])/(normalizer[ii][1] - normalizer[ii][0])
        elif todo=='max*2+1':
            if donorm: normalizer[ii] = np.max(np.abs(vec))
            data[:,ii]  =   vec / 2 / normalizer[ii] + 0.5
    """            
    # Center the data
    if mu is None:
        mu      =   np.mean(data, axis=0)
    if sigma is None:
        sigma   =   np.std(data, axis=0)
    data        =   (data - np.tile(mu, [len(data), 1])) / np.tile(sigma, [len(data), 1])
    """
    return data, normalizer


def do_reduce_data(X, pca=None, mu=None, sigma=None, nComp=None):
    """
    # Remove NAN - by the way this should be fixed upfront
    x, y        =   np.where(X.isnull())
    x           =   np.unique(x)
    annInput    =   np.delete(annInput, (x), axis=0)
    annTarget   =   np.delete(annTarget, (x), axis=0)
    indices     =   np.delete(indices, (x), axis=0)
    # Center the data
    if mu is None:
        mu      =   np.mean( annInput.astype(float), axis=0 )
    if sigma is None:
        sigma   =   np.std( annInput.astype(float), axis=0 )
    annInput    =   ( annInput - np.tile(mu, [len(annInput),1]) ) / np.tile(sigma, [len(annInput),1])
    """
    # Perform PCA - just look for nb of components
    if pca is None:
        if nComp is None:
            pca     =   PCA(svd_solver='full', whiten=True)
            pca.fit(X)
            nComp   =   ut_cumsum_thresh(pca.explained_variance_, 0.95)
        # Perform PCA - transform the data
        pca     =   PCA(n_components=nComp, svd_solver='full', whiten=True)
        pca.fit(X)
    annInput    =   pca.transform(X)
    return annInput, pca


def do_ANN_training(repoPSt, repoPbP, repoCode, repoModel):
    # --- GET TRAINING DATASET
    # List non-lockout seasons
    allS_p          =   ut_find_folders(repoPbP, True)
    """
    # Get data
    X,Y, X_all,POS_all,colNm=   get_training_data(allS_p)
    with open( path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/trainingData.p'), 'wb') as f:
        pickle.dump({'X':X, 'Y':Y, 'X_all':X_all, 'colNm':colNm, 'POS_all':POS_all}, f)
    """
    with open( path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/trainingData.p'), 'rb') as f:
        DT      =   pickle.load(f)
        colNm   =   DT['colNm']
        X       =   DT['X'][colNm]
        Y       =   DT['Y']
        X_all   =   DT['X_all']
        POS_all =   DT['POS_all']


    # --- PRE-PROCESS DATA
    Y, X, POS_all, X_all =   ut_sanitize_matrix(Y, X), ut_sanitize_matrix(X), ut_sanitize_matrix(POS_all, X_all), ut_sanitize_matrix(X_all)
    X_all_S, Nrm=   do_normalize_data(X_all[(POS_all!='D').values])
    X_S, _      =   do_normalize_data(X, normalizer=Nrm)
    _, pca      =   do_reduce_data(X_all_S, nComp=18)
    X_S_P, _    =   do_reduce_data(X_S, pca=pca, nComp=18)
    # --- BUILD THE NETWORK
    nNodes  =   [X_S_P.shape[1], 15, Y.shape[1]]
    CLS     =   ANN_classifier( deepcopy(nNodes) )
    # --- TRAIN THE NETWORK
    nIter   =   50
    CLS.ann_train_network(nIter, X_S_P, Y.values, svname=repoModel)
    # --- DISPLAY NETWORK ACCURACY
    #CLS.ann_display_accuracy()
    return Nrm, pca, colNm


def do_ANN_classification(repoModel, dtCols, normalizer, pca, upto='2016-07-01', asof='2015-09-01', nGames=80):
    # --- RETRIEVE DATA
    DT, _       =   pull_stats(repoPSt, repoPbP, upto=upto, asof=asof, nGames=nGames)     #   sea_pos,
    # --- PRE-PROCESS DATA
    DT[dtCols]  =   ut_sanitize_matrix(DT[dtCols])
    DT_n, _     =   do_normalize_data(DT[dtCols], normalizer=normalizer)
    DT_n_p, _   =   do_reduce_data(DT_n, pca=pca)
    #annI, annT, _, _, _     =   do_process_data( DT[dtCols], pd.DataFrame(np.zeros([len(DT), 1])), pca=pca, mu=mu, sigma=sigma )
    # --- RELOAD THE NETWORK IMAGE
    CLS         =   ANN_classifier([DT_n_p.shape[1], 20, 2])
    CLS.ann_reload_network(repoModel)
    # --- CLASSIFY DATA
    DTfeed      =   {CLS.annX:DT_n_p}
    return DT, pd.DataFrame(DT_n_p, index=DT.index), CLS.sess.run(CLS.annY, feed_dict=DTfeed)


def do_clustering(data, classes, upto, root):

    # Make constraints
    years       =   ''.join([str(int(x)) for x in list((int(upto.split('-')[0]) * np.ones([1, 2]) - np.array([1, 0]))[0])])
    selke       =   to_pandas_selke( path.join(root, 'Databases/Hockey/PlayerStats/raw/' + years + '/trophy_selke_nominees.csv') )
    selke_id    =   [list(data.index).index(x) for x in selke[selke['Pos'] != 'D'].index]
    ross        =   to_pandas_ross( path.join(root, 'Databases/Hockey/PlayerStats/raw/' + years + '/trophy_ross_nominees.csv') )
    ross_id     =   [list(data.index).index(x) for x in ross[ross['pos'] != 'D'].index]

    # --- Clean constraints
    # Remove duplicates
    torem       =   list(set(ross_id).intersection(selke_id))
    maxV        =   np.argmax(classes.iloc[torem].values, axis=1).astype(bool)
    selke_id    =   ut_difference(selke_id, list(compress(torem, maxV)))
    selke_wgt   =   selke['WEIGHT'].values / np.max(selke['WEIGHT'].values)
    ross_id     =   ut_difference(ross_id, list(compress(torem, maxV != True)))
    ross_wgt    =   ross['WEIGHT'].values / np.max(ross['WEIGHT'].values)

    # Get poorest ranked players
    seed        =   classes.min(axis=0)
    distance    =   np.sqrt(((classes - seed) ** 2).sum(axis=1)).sort_values()
    poor_id     =   [classes.index.get_loc(x) for x in distance.index[:30]]
    poor_id     =   ut_difference(ut_difference(poor_id, selke_id), ross_id)
    constraints =   ut_make_constraints((selke_id, selke_wgt), (ross_id, ross_wgt), poor_id)
    constraints =   pd.DataFrame(constraints)
    constraints =   constraints[constraints[0] != constraints[1]]

    # Make clusters
    cls_data    =   list(list(x) for x in classes.values)
    ml, cl, dmp =   [], [], []
    [ml.append(tuple(x[:2].astype('int'))) if x[-1] > 0.5 else dmp.append(tuple(x[:2])) for x in constraints.values]
    [cl.append(tuple(x[:2].astype('int'))) if x[-1] < -0.5 else dmp.append(tuple(x[:2])) for x in constraints.values]

    clusters, centers, cost = cop_kmeans(cls_data, 3, ml, cl, max_iter=3000000, tol=1e+5, initialization='hockey')
    #display_clustering(classes, clusters, ross_id, selke_id)
    return clusters, centers, selke_id, ross_id


def get_data_for_clustering(repoModel, dtCols, normalizer, pca, upto='2016-07-01', asof='2015-09-01', nGames=80):
    DT, DT_n, pCl   =   do_ANN_classification(repoModel, dtCols, normalizer, pca, upto=upto, asof=asof, nGames=nGames)
    pCl         =   pd.DataFrame(pCl, index=DT.index, columns=['OFF', 'DEF'])
    # Filter players - keep forwards only
    isForward   =   [x != 'D' for x in DT['position'].values]
    isRegular   =   [x > int(nGames * .05) for x in DT['gmPl']]
    filter      =   [True if x and y else False for x, y in zip(isForward, isRegular)]
    fwd_dt      =   DT[filter]
    fwd_cl      =   pCl[filter]  # This is the clustering matrix
    return fwd_dt, fwd_cl


def do_clustering_multiyear(repoModel, dtCols, normalizer, pca, root):
    # Make constraints
    allS_p      =   ut_find_folders(repoPbP, True)
    years       =   [[x.split('_')[1][:4], x.split('_')[1][4:]] for x in allS_p]
    count       =   0
    allCla      =   pd.DataFrame()
    allCON      =   pd.DataFrame()
    allCls      =   []
    allSLK      =   []
    allROS      =   []

    ###### ECLUDE YEAR 2003-2004 : PROBLEM WITH FREDRIK MODIN'S DATA - NAMED AS FREDDY MODIN IN THE NHL STATS PAGE
    years.pop( years.index(['2003', '2004']) )
    all_centers =   []
    for iy in years:
        # Get data
        data, classes   =   get_data_for_clustering(repoModel, dtCols, normalizer, pca, upto=iy[1]+'-07-01', asof=iy[0]+'-09-01', nGames=81)
        # Get trophy nominees
        selke       =   to_pandas_selke( path.join(root, 'Databases/Hockey/PlayerStats/raw/' + ''.join(iy) + '/trophy_selke_nominees.csv') )
        selke       =   selke[~selke.index.duplicated(keep='first')]
        selke_id    =   [list(data.index).index(x) for x in selke[selke['Pos'] != 'D'].index]
        ross        =   to_pandas_ross( path.join(root, 'Databases/Hockey/PlayerStats/raw/' + ''.join(iy) + '/trophy_ross_nominees.csv') )
        ross        =   ross[~ross.index.duplicated(keep='first')]
        ross_id     =   [list(data.index).index(x) for x in ross[ross['pos'] != 'D'].index]
        # --- Clean constraints
        # Remove duplicates
        torem       =   list( set(ross_id).intersection(selke_id) )
        maxV        =   np.argmax(classes.iloc[torem].values, axis=1).astype(bool)
        selke_id    =   ut_difference( selke_id, list( compress(torem, maxV) ))
        selke_wgt   =   selke.loc[data.iloc[selke_id].index]['WEIGHT'].values
        selke_wgt   =   selke_wgt/np.max(selke_wgt)
        ross_id     =   ut_difference( ross_id, list( compress(torem, maxV!=True) ))
        ross_wgt    =   ross.loc[data.iloc[ross_id].index]['WEIGHT'].values
        ross_wgt    =   ross_wgt / np.max(ross_wgt)
        # Get poorest ranked players
        seed        =   classes.min(axis=0)
        distance    =   np.sqrt( ((classes - seed)**2).sum(axis=1) ).sort_values()
        poor_id     =   [classes.index.get_loc(x) for x in distance.index[:30]]
        poor_id     =   ut_difference( ut_difference( poor_id, selke_id ), ross_id )
        # Make the constraints
        constraints =   ut_make_constraints( (selke_id, selke_wgt), (ross_id, ross_wgt), poor_id )
        constraints =   pd.DataFrame(constraints)
        constraints =   constraints[constraints[0] != constraints[1]]

        # Make clusters
        cls_data        =   list(list(x) for x in classes.values)
        cOm             =   [list(ut_center_of_mass(classes.iloc[x].values, np.reshape(y, [-1, 1]))) for x, y in zip([selke_id, ross_id, poor_id], [selke_wgt, ross_wgt, np.ones([1, len(poor_id)])])]
        ml, cl, dmp     = [], [], []
        [ml.append(tuple(x[:2].astype('int'))) if x[-1] > 0.5 else dmp.append(tuple(x[:2])) for x in constraints.values]
        [cl.append(tuple(x[:2].astype('int'))) if x[-1] < -0.5 else dmp.append(tuple(x[:2])) for x in constraints.values]
        clusters, centers, cost   =   cop_kmeans(cls_data, 3, ml, cl, max_iter=1000, tol=1e-4, initialization=cOm)
        all_centers.append(centers)

        # Append
        allSLK  =   allSLK + list(np.add(selke_id, len(allCla)))
        allROS  =   allROS + list(np.add(ross_id, len(allCla)))
        allCla  =   pd.concat((allCla, classes), axis=0)
        allCls  =   allCls + clusters
        allCtr  =   [list(x) for x in np.mean(np.array(all_centers),axis=0)]
        #display_clustering(classes, clusters, centers, ross_id, selke_id)
    #print('year: ', iy, 'cost: ', np.sum(cost))
    #display_clustering(allCla, allCls, allCtr, allROS, allSLK)

    # Cluster the centers
    all_centers     =   np.concatenate( np.array(all_centers), axis=0 )
    all_centers     =   list([list(x) for x in all_centers])
    index           =   np.array(range( int(len(all_centers)/3) ))*3
    constraints     =   ut_make_constraints( (list(index), list(np.ones([len(index),1]))), (list(index+1), list(np.ones([len(index),1]))), list(index+2))
    constraints     =   pd.DataFrame(constraints)
    constraints     =   constraints[constraints[0] != constraints[1]]
    ml, cl, dmp     =   [], [], []
    [ml.append(tuple(x[:2].astype('int'))) if x[-1] > 0.5 else dmp.append(tuple(x[:2])) for x in constraints.values]
    [cl.append(tuple(x[:2].astype('int'))) if x[-1] < -0.5 else dmp.append(tuple(x[:2])) for x in constraints.values]
    cOm             =   [list(ut_center_of_mass(classes.iloc[x].values, np.reshape(y, [-1,1]) )) for x,y in zip([selke_id, ross_id, poor_id], [selke_wgt, ross_wgt, np.ones([1, len(poor_id)])])]
    glCL, glCT, _   =   cop_kmeans(all_centers, 3, ml, cl, max_iter=1000, tol=1e-4, initialization='hockey')
    display_clustering(pd.DataFrame(all_centers, columns=['OFF', 'DEF']), glCL, glCT, list(index), list(index+1))
    # Relate centers to trophies
    iSlk            =   np.argmin( [np.sqrt(np.sum(np.subtract([0,1], x)**2)) for x in glCT] )
    iRoss           =   np.argmin( [np.sqrt(np.sum(np.subtract([1,0], x)**2)) for x in glCT] )
    iPoor           =   list( set(range(3)).difference([iSlk,iRoss]) )[0]
    global_centers  =   {'selke':glCT[iSlk], 'ross':glCT[iRoss], 'poor':glCT[iPoor]}
    # Save result
    pickle.dump({'global_centers':global_centers, 'normalizer':normalizer, 'pca':pca, 'dtCols':dtCols}, \
                open(path.join(repoModel, 'baseVariables.p'), 'wb') )
    return global_centers


def display_clustering(classification, clusters, centers, ross_id, selke_id):
    Fig     =   plt.figure()
    # Display ground truth
    Ax1     =   Fig.add_subplot(121)
    Ax1.scatter(classification['OFF'], classification['DEF'])
    Ax1.scatter(classification.iloc[selke_id]['OFF'].values, classification.iloc[selke_id]['DEF'].values, color='k')
    Ax1.scatter(classification.iloc[ross_id]['OFF'].values, classification.iloc[ross_id]['DEF'].values, color='r')
    Ax1.legend(['Not nominated', 'Selke ground truth (def)', 'Art Ross ground truth (off)'])
    Ax1.set_xlabel('likelihood to win offensive trophy')
    Ax1.set_ylabel('likelihood to win defensive trophy')
    # allP    =   [plt.text(fwd_cl.iloc[x]['OFF'], fwd_cl.iloc[x]['DEF'], list(fwd_dt.index)[x]) for x in range(len(fwd_dt))]
    # Display constrained clustering
    Ax2     =   Fig.add_subplot(122)
    Ax2.scatter(classification['OFF'], classification['DEF'], c=clusters)
    Ax2.scatter([x[0] for x in centers], [x[1] for x in centers], marker="D", c=[0,1,2])
    Ax2.set_xlabel('likelihood to win offensive trophy')
    Ax2.set_ylabel('likelihood to win defensive trophy')
    return Fig, Ax1, Ax2


def do_assess_clustering_robustness(dtCols, normalizer, global_centers, pca, nGames=80):
    # This function computes the confusion of the global classification
    # List years
    allS_p      =   ut_find_folders(repoPbP, True)
    years       =   [[x.split('_')[1][:4], x.split('_')[1][4:]] for x in allS_p]
    years.pop(years.index(['2003', '2004']))
    nGamesL     =   list( range(10,81,10) )
    accuracy    =   pd.DataFrame(np.zeros([len(years), 8]), columns=nGamesL, index=[''.join(x) for x in years])
    for iy in years:
        for nG in nGamesL:
            # Retrieve data
            data, classes   =   get_data_for_clustering(repoModel, dtCols, normalizer, pca, upto=iy[1]+'-07-01', asof=iy[0]+'-09-01', nGames=nG)
            # Get trophy nominees
            selke       =   to_pandas_selke(path.join(root, 'Databases/Hockey/PlayerStats/raw/' + ''.join(iy) + '/trophy_selke_nominees.csv'))
            selke       =   selke[~selke.index.duplicated(keep='first')]
            selke_id    =   [list(data.index).index(x) for x in selke[selke['Pos'] != 'D'].index]
            ross        =   to_pandas_ross(path.join(root, 'Databases/Hockey/PlayerStats/raw/' + ''.join(iy) + '/trophy_ross_nominees.csv'))
            ross        =   ross[~ross.index.duplicated(keep='first')]
            ross_id     =   [list(data.index).index(x) for x in ross[ross['pos'] != 'D'].index]
            seed        =   classes.min(axis=0)
            distance    =   np.sqrt(((classes - seed) ** 2).sum(axis=1)).sort_values()
            poor_id     =   [classes.index.get_loc(x) for x in distance.index[:30]]
            poor_id     =   ut_difference(ut_difference(poor_id, selke_id), ross_id)
            # --- Compute confusion
            # Selke players
            dst_slk     =   np.sqrt( np.sum(np.subtract(classes.iloc[selke_id].values, global_centers['selke'])**2, axis=1) )
            dst_ross    =   np.sqrt( np.sum(np.subtract(classes.iloc[selke_id].values, global_centers['ross']) ** 2, axis=1))
            dst_poor    =   np.sqrt( np.sum(np.subtract(classes.iloc[selke_id].values, global_centers['poor']) ** 2, axis=1))
            slk_min     =   np.argmin( np.array([dst_slk, dst_ross, dst_poor]), axis=0 )
            # Ross players
            dst_slk     =   np.sqrt(np.sum(np.subtract(classes.iloc[ross_id].values, global_centers['selke']) ** 2, axis=1))
            dst_ross    =   np.sqrt(np.sum(np.subtract(classes.iloc[ross_id].values, global_centers['ross']) ** 2, axis=1))
            dst_poor    =   np.sqrt(np.sum(np.subtract(classes.iloc[ross_id].values, global_centers['poor']) ** 2, axis=1))
            ros_min     =   np.argmin(np.array([dst_slk, dst_ross, dst_poor]), axis=0)
            # Poor players
            dst_slk     =   np.sqrt(np.sum(np.subtract(classes.iloc[poor_id].values, global_centers['selke']) ** 2, axis=1))
            dst_ross    =   np.sqrt(np.sum(np.subtract(classes.iloc[poor_id].values, global_centers['ross']) ** 2, axis=1))
            dst_poor    =   np.sqrt(np.sum(np.subtract(classes.iloc[poor_id].values, global_centers['poor']) ** 2, axis=1))
            por_min     =   np.argmin(np.array([dst_slk, dst_ross, dst_poor]), axis=0)
            # Make matrix
            MTX         =   [[np.sum(x==0)/len(x), np.sum(x==1)/len(x), np.sum(x==2)/len(x)] for x in [slk_min, ros_min, por_min]]
            accuracy.loc[''.join(iy)][nG]    =   np.sum(np.diag(MTX))/np.sum(MTX)
    return accuracy


# LAUNCHER:
# =========
root        =   '/home/younesz/Documents'
#root        =   '/Users/younes_zerouali/Documents/Stradigi'
repoPbP     =   path.join(root, 'Databases/Hockey/PlayByPlay')
repoPSt     =   path.join(root, 'Databases/Hockey/PlayerStats/player')
repoRaw     =   path.join(root, 'Databases/Hockey/PlayerStats/raw')
repoCode    =   path.join(root, 'Code/Python')
repoModel   =   path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')



"""
# ============================================
# === MAKE THE PLAYER CLASSIFICATION FRAMEWORK

# Train automatic classifier - ANN
normalizer, pca, dtCols     =   do_ANN_training(repoPSt, repoPbP, repoCode, repoModel)     # Nrm is the normalizing terms for the raw player features
# Classify player data : MULTIPLE YEARS
global_centers  =   do_clustering_multiyear(repoModel, dtCols, normalizer, pca, root)
# ============================================
"""


"""
# ============================================
# === ASSESS ROBUSTNESS OF THE CLASSIFICATION FRAMEWORK

# Reload pre-saved clustering
VAR         =   pickle.load( open(path.join(repoModel, 'baseVariables.p'), 'rb') )
dtCols, normalizer, global_centers, pca     =   VAR['dtCols'], VAR['normalizer'], VAR['global_centers'], VAR['pca']
get_data_for_clustering(repoModel, dtCols, normalizer, pca)
# Assess robustness
accuracy    =   do_assess_clustering_robustness()
# ============================================


"""

#



"""

# --- Classify player data : SINGLE TIME SLOT
upto, asof, nGames  =   '2011-07-01', '2010-09-01', 80
data, classes       =   get_data_for_clustering(repoModel, dtCols, normalizer, pca, upto=upto, asof=asof, nGames=nGames)
# Apply constrained clustering
clusters, centers, selke_id, ross_id    =   do_clustering(data, classes, upto, root)
display_clustering(classes, clusters, centers, ross_id, selke_id)










Xs, Ys      =   do_prep_data( ut_find_folders(repoPbP, True) )
#Xs, Ys     =   do_prep_data(['Season_20112012'])
#Xsp, Ysp, Ind   =   do_process_data(Xs, Ys, nComp=18)

# Make classification - manual
upto    =   '2013-07-01'
nGames  =   80
PLclass, PLnames    =   do_manual_classification(repoPSt, repoPbP, upto, nGames)

# Sanity check
season  =   '20122013'
validate_classes(PLclass, PLnames, repoRaw, season)

PQuart_off.index[CLASS==0]





plNames     =   get_player_names(repoPbP)
all_dt      =   {}
for pl in plNames:
    with open(path.join(repoPSt, pl.replace(' ', '_')+'.p'), 'rb') as f:
        all_dt[pl]  =   pickle.load(f)
with open(path.join(repoPSt, 'all_players.p'), 'wb') as f:
        pickle.dump(all_dt, f)
"""