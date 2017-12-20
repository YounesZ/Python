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
from Utils.programming.ut_find_folders import *
from sklearn.model_selection import train_test_split
from Utils.programming.ut_sanitize_matrix import ut_sanitize_matrix
from Utils.scraping.convert_raw import get_player_names
from Utils.scraping.convert_trophies import to_pandas_selke, to_pandas_ross
from Clustering.copkmeans.cop_kmeans import cop_kmeans



class ANN_classifier():

    def __init__(self, nNodes):
        # Launch the builder
        nInputs     =   nNodes.pop(0)
        nOutputs    =   nNodes.pop(-1)
        nHidden     =   len(nNodes)
        self.ann_build_network(nInputs, nOutputs, nNodes)

    def ann_build_network(self, nInputs, noutputs, nNodes):
        # Architecture - 1 layer
        self.annX   =   tf.placeholder(tf.float32, [None, nInputs], name='Input_to_the_network-player_features')
        self.annY_  =   tf.placeholder(tf.float32, [None, 2], name='Ground_truth')
        annW1       =   tf.Variable(tf.truncated_normal([nInputs, nNodes[0]], stddev=0.1))
        annB1       =   tf.Variable(tf.ones([1, nNodes[0]]) / 10)
        Y1          =   tf.nn.relu(tf.matmul(self.annX, annW1) + annB1)
        annW2       =   tf.Variable(tf.truncated_normal([nNodes[0], noutputs], stddev=0.1))
        annB2       =   tf.Variable(tf.ones([1, noutputs]) / 10)
        self.annY   =   tf.matmul(Y1, annW2) + annB2
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
        saver   =   tf.train.Saver()
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
            save_path       =   saver.save(self.sess, svname)
            self.mirror     =   svname

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

    def ann_reload_network(self, image):
        # Init saver
        saver       =   tf.train.Saver()
        # Initialize the variables
        self.sess   =   tf.Session()
        init        =   tf.global_variables_initializer()
        self.sess.run(init)
        # Restore model weights from previously saved model
        saver.restore(self.sess, image)


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
    with open(path.join(repoPSt, 'all_players.p'), 'rb') as f:
        all_pl  =   pickle.load(f)
    for pl in plNames:
        # Load stats file
        plStat          =   all_pl[pl]
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


def get_training_data(season):
    X_train     =   pd.DataFrame()
    Y_train     =   pd.DataFrame()
    X_all       =   pd.DataFrame()
    # Loop on seasons and collect data
    for isea in season:
        # Get end time stamp
        sea_name=   isea.replace('Season_', '')
        sea_strt=   sea_name[:4] + '-09-01'
        sea_end =   sea_name[-4:] + '-07-01'
        # Pull stats
        sea_stats, sea_pos, sea_numeric, sea_normal = pull_stats(repoPSt, repoPbP, sea_strt, sea_end)
        dtCols  =   list(set(sea_numeric).union(sea_normal))
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
    return X_train, Y_train, X_all


def do_normalize_data(data, mu=None, sigma=None, normalizer=None):
    # Custom normalization
    wtd     =   ['max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max*2+1', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'minmax', 'minmax', 'minmax', 'max', 'max', 'max', 'max', 'minmax', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'minmax']
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


def do_reduce_data(X, Y, pca=None, mu=None, sigma=None):
    # Data
    annInput    =   deepcopy(X.values)
    annTarget   =   deepcopy(Y.values)
    indices     =   X.index
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
    # Perform PCA - just look for nb of components
    if pca is None:
        pca     =   PCA(svd_solver='full', whiten=True)
        pca.fit(annInput)
        nComp   =   ut_cumsum_thresh(pca.explained_variance_, 0.95)
        # Perform PCA - transform the data
        pca     =   PCA(n_components=nComp, svd_solver='full', whiten=True)
        pca.fit(annInput)
    annInput    =   pca.transform(annInput)
    return annInput, annTarget, pca, mu, sigma


def do_ANN_training(repoPSt, repoPbP, repoCode):
    # --- GET TRAINING DATASET
    """
    # List non-lockout seasons
    allS_p          =   ut_find_folders(repoPbP, True)
    # Get data
    X,Y, X_all      =   get_training_data(allS_p)
    colNm           =   X.columns
    with open( path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/trainingData.p'), 'wb') as f:
        pickle.dump({'X':X, 'Y':Y, 'X_all':X_all, 'colNm':X.columns}, f)
    """
    with open( path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/trainingData.p'), 'rb') as f:
        DT      =   pickle.load(f)
        colNm   =   DT['colNm']
        X       =   DT['X'][colNm]
        Y       =   DT['Y']
        X_all   =   DT['X_all']
    # --- PRE-PROCESS DATA
    Y, X, X_all =   ut_sanitize_matrix(Y, X), ut_sanitize_matrix(X), ut_sanitize_matrix(X_all)
    X_all_S, Nrm=   do_normalize_data(X_all)
    X_S, _      =   do_normalize_data(X, normalizer=Nrm)
    # --- BUILD THE NETWORK
    nNodes  =   [X_S.shape[1], 40, Y.shape[1]]
    CLS     =   ANN_classifier(nNodes)
    # --- TRAIN THE NETWORK
    nIter   =   50
    """
    netname =   'MODEL_perceptron_1layer_10units_relu/model.ckpt'
    svname  =   path.join('/home/younesz/Documents/Code/Python/ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification', netname)
    """
    CLS.ann_train_network(nIter, X_S, Y.values)
    # --- DISPLAY NETWORK ACCURACY
    #CLS.ann_display_accuracy()
    return CLS, Nrm, colNm


def do_ANN_classification(upto, nGames, dtCols, normalizer, CLS):
    # --- RETRIEVE DATA
    DT, plPos, numC, nrmC   =   pull_stats(repoPSt, repoPbP, upto=upto, nGames=nGames)     #   sea_pos,
    # --- PRE-PROCESS DATA
    plPos, DT   =   ut_sanitize_matrix(plPos, DT[dtCols]), ut_sanitize_matrix(DT[dtCols])
    DT_n, _     =   do_normalize_data(DT[dtCols], normalizer=normalizer)
    #annI, annT, _, _, _     =   do_process_data( DT[dtCols], pd.DataFrame(np.zeros([len(DT), 1])), pca=pca, mu=mu, sigma=sigma )
    # --- RELOAD THE NETWORK IMAGE
    # CLS.ann_reload_network(mirror)
    # --- CLASSIFY DATA
    DTfeed      =   {CLS.annX:DT_n}
    return DT, pd.DataFrame(DT_n, index=DT.index, columns=dtCols), plPos, CLS.sess.run(CLS.annY, feed_dict=DTfeed)


def do_clustering(data, classes, upto, root):
    # Make constraints
    years       =   ''.join([str(int(x)) for x in list((int(upto.split('-')[0]) * np.ones([1, 2]) - np.array([1, 0]))[0])])
    selke       =   to_pandas_selke( path.join(root, 'Databases/Hockey/PlayerStats/raw/' + years + '/trophy_selke_nominees.csv') )
    selke_id    =   [data.index.tolist().index(x) for x in selke[selke['Pos'] != 'D'].index]
    ross        =   to_pandas_ross( path.join(root, 'Databases/Hockey/PlayerStats/raw/' + years + '/trophy_ross_nominees.csv') )
    ross_id     =   [data.index.tolist().index(x) for x in ross[ross['pos'] != 'D'].index]

    # --- Clean constraints
    # Remove duplicates
    torem       =   list( set(ross_id).intersection(selke_id) )
    maxV        =   np.argmax(classes.iloc[torem].values, axis=1).astype(bool)
    selke_id    =   list( set(selke_id).difference(list( compress(torem, maxV) )) )
    ross_id     =   list( set(ross_id).difference(list( compress(torem, maxV!=True) )) )
    # Get poorest ranked players
    seed        =   classes.min(axis=0)
    distance    =   np.sqrt( ((classes - seed)**2).sum(axis=1) ).sort_values()
    poor_id     =   [classes.index.get_loc(x) for x in distance.index[:30]]
    poor_id     =   list( set(poor_id).difference(selke_id).difference(ross_id) )
    constraints =   ut_make_constraints(selke_id, ross_id, poor_id)
    constraints =   pd.DataFrame(constraints)
    constraints =   constraints[constraints[0] != constraints[1]]

    # Make clusters
    cls_data    =   list(list(x) for x in classes.values)
    ml, cl      =   [], []
    [ml.append(tuple(x[:2])) if x[-1] == 1 else cl.append(tuple(x[:2])) for x in constraints.values]
    clusters, centers = cop_kmeans(cls_data, 3, ml, cl, max_iter=1000, tol=1e-4)
    return clusters, centers, selke_id, ross_id


def get_data_for_clustering(upto, nGames, dtCols, normalizer, CLS):
    DT, DT_n, pPos, pCl   =   do_ANN_classification(upto, nGames, dtCols, normalizer, CLS)
    pCl         =   pd.DataFrame(pCl, index=DT.index, columns=['OFF', 'DEF'])
    # Filter players - keep forwards only
    isForward   =   [x != 'D' for x in pPos.values]
    isRegular   =   [x > int(nGames * .75) for x in DT['gamesPlayed']]
    filter      =   [True if x and y else False for x, y in zip(isForward, isRegular)]
    fwd_dt      =   DT[filter]
    fwd_cl      =   pCl[filter]  # This is the clustering matrix
    return fwd_dt, fwd_cl


def do_clustering_multiyear(data, classes, root):
    # Make constraints
    allS_p      =   ut_find_folders(repoPbP, True)
    years       =   [[x.split('_')[1][:4], x.split('_')[1][4:]] for x in allS_p]
    for iy in years:
        selke       =   to_pandas_selke( path.join(root, 'Databases/Hockey/PlayerStats/raw/' + ''.join(iy) + '/trophy_selke_nominees.csv') )
        selke_id    =   [data.index.tolist().index(x) for x in selke[selke['Pos'] != 'D'].index]
        ross        =   to_pandas_ross( path.join(root, 'Databases/Hockey/PlayerStats/raw/' + years + '/trophy_ross_nominees.csv') )
        ross_id     =   [data.index.tolist().index(x) for x in ross[ross['pos'] != 'D'].index]

    # --- Clean constraints
    # Remove duplicates
    torem       =   list( set(ross_id).intersection(selke_id) )
    maxV        =   np.argmax(classes.iloc[torem].values, axis=1).astype(bool)
    selke_id    =   list( set(selke_id).difference(list( compress(torem, maxV) )) )
    ross_id     =   list( set(ross_id).difference(list( compress(torem, maxV!=True) )) )
    # Get poorest ranked players
    seed        =   classes.min(axis=0)
    distance    =   np.sqrt( ((classes - seed)**2).sum(axis=1) ).sort_values()
    poor_id     =   [classes.index.get_loc(x) for x in distance.index[:30]]
    poor_id     =   list( set(poor_id).difference(selke_id).difference(ross_id) )
    constraints =   ut_make_constraints(selke_id, ross_id, poor_id)
    constraints =   pd.DataFrame(constraints)
    constraints =   constraints[constraints[0] != constraints[1]]

    # Make clusters
    cls_data    =   list(list(x) for x in classes.values)
    ml, cl      =   [], []
    [ml.append(tuple(x[:2])) if x[-1] == 1 else cl.append(tuple(x[:2])) for x in constraints.values]
    clusters, centers = cop_kmeans(cls_data, 3, ml, cl, max_iter=1000, tol=1e-4)
    return clusters, centers, selke_id, ross_id


def display_clustering(classification, clusters, ross_id, selke_id):
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
    Ax2.set_xlabel('likelihood to win offensive trophy')
    Ax2.set_ylabel('likelihood to win defensive trophy')
    return Fig, Ax1, Ax2


# LAUNCHER:
# =========
#root        =   '/home/younesz/Documents'
root        =   '/Users/younes_zerouali/Documents/Stradigi'
repoPbP     =   path.join(root, 'Databases/Hockey/PlayByPlay')
repoPSt     =   path.join(root, 'Databases/Hockey/PlayerStats/player')
repoRaw     =   path.join(root, 'Databases/Hockey/PlayerStats/raw')
repoCode    =   path.join(root, 'Python')


"""
# Train automatic classifier - ANN
CLS, normalizer, dtCols     =   do_ANN_training(repoPSt, repoPbP, repoCode)     # Nrm is the normalizing terms for the raw player features

# --- Classify player data
upto, nGames            =   '2010-07-01', 80
pl_data, pl_classes     =   get_data_for_clustering(upto, nGames, dtCols, normalizer, CLS)
# Apply constrained clustering
clusters, centers, selke_id, ross_id    =   do_clustering(pl_data, pl_classes, upto, root)
display_clustering(pl_classes, clusters, ross_id, selke_id)







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