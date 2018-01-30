import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sys import stdout
from copy import deepcopy
from random import shuffle
from Utils.programming.ut_find_folders import *
from Utils.programming.ut_sanitize_matrix import ut_sanitize_matrix
from ReinforcementLearning.NHL.playbyplay.agent import Agent
from ReinforcementLearning.NHL.playerstats.nhl_player_stats import pull_stats, do_normalize_data, do_reduce_data, ANN_classifier


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
            iSea.list_game_ids( self.repoPbP, self.repoPSt )
            games_lst   =   pd.concat( (games_lst, iSea.games_id), axis=0 )
        self.games_lst  =   games_lst


    def pull_RL_data(self, repoModel, repoSave=None):
        # Prepare players model: reload info
        self.players_model  =   pickle.load(open(path.join(repoModel, 'baseVariables.p'), 'rb'))
        self.classifier     =   {'sess':tf.Session(), 'annX':[], 'annY':[]}
        saver               =   tf.train.import_meta_graph(path.join(repoModel, path.basename(repoModel) + '.meta'))
        graph               =   self.classifier['sess'].graph
        self.classifier['annX'] =   graph.get_tensor_by_name('Input_to_the_network-player_features:0')
        self.classifier['annY'] =   graph.get_tensor_by_name('prediction:0')
        saver.restore(self.classifier['sess'], tf.train.latest_checkpoint(path.join(repoModel, './')))
        # Make lines dictionary
        self.make_line_dictionary()
        # List line shifts
        RL_data     =   pd.DataFrame()
        GAME_data   =   pd.DataFrame()
        PLAYER_data =   pd.DataFrame()
        count       =   0
        allR        =   []
        for iy,ic,ih,ia in zip(self.games_lst['season'].values,self.games_lst['gcode'].values,self.games_lst['hometeam'].values,self.games_lst['awayteam'].values):
            # Extract state-space
            iGame       =   Game(self.repoPbP, self.repoPSt, iy, ic)

            # Check if some data was retrieved:
            if len(iGame.df_wc)>0:
                iGame.pull_line_shifts('both', minduration=20)
                iGame.pick_regulartime()
                iGame.pick_equalstrength()
                iGame.pull_players_classes(self.players_model, self.classifier)
                # Add game identifier data
                iGame.lineShifts['season']      =   iy
                iGame.lineShifts['gameCode']    =   ic
                iGame.lineShifts['hometeam']    =   ih
                iGame.lineShifts['awayteam']    =   ia
                # Check if some data was retrieved:
                if len(iGame.player_classes)>0:
                    S, A, R, nS, nA, coded      =   iGame.build_statespace(self.line_dictionary)
                    allR.append( np.sum(R) )
                    # Concatenate data
                    df_ic       =   np.transpose(np.reshape(np.concatenate((S, A, R)), [3, -1]))
                    RL_data     =   pd.concat((RL_data, pd.DataFrame(df_ic, columns=['state', 'action', 'reward'])), axis=0)
                    GAME_data   =   pd.concat((GAME_data, iGame.lineShifts[coded]), axis=0)
                    # Players data
                    plDT        =   iGame.player_classes
                    plDT['season']  =   iy
                    plDT['gameCode']=   ic
                    PLAYER_data     =   pd.concat((PLAYER_data, plDT), axis=0)
                    # Save data
                    if not repoSave is None and count % 20 == 0:
                        pickle.dump({'RL_data': RL_data, 'nStates': nS, 'nActions': nA}, open(path.join(repoSave, 'RL_teaching_data.p'), 'wb'))
                        pickle.dump(GAME_data, open(path.join(repoSave, 'GAME_data.p'), 'wb'))
                        pickle.dump(PLAYER_data, open(path.join(repoSave, 'PLAYER_data.p'), 'wb') )
                else:
                    print('*** EMPTY GAME ***')
            else:
                print('*** EMPTY GAME ***')

            # Status bar
            stdout.write('\r')
            # the exact output you're looking for:
            stdout.write("Game %i/%i - season %s game %s: [%-60s] %d%%, completed" % (count, len(self.games_lst), iy, ic, '=' * int(count / len(self.games_lst) * 60), 100 * count / len(self.games_lst)))
            stdout.flush()
            count   +=  1
        self.RL_data        =   RL_data
        self.state_size     =   nS
        self.action_size    =   nA


    def teach_RL_agent(self):
        # Instantiate the agent
        agent       =   Agent(self.state_size, self.action_size)
        # --- TEACH THE AGENT
        # List all samples
        iSamples    =   list( range(self.RL_data.shape[0]) )
        shuffle(iSamples)
        count       =   0
        # Loop on samples and teach
        for iS in iSamples:
            # Get new teaching example
            S,A,R   =   self.RL_data.iloc[iS]['state'], self.RL_data.iloc[iS]['action'], self.RL_data.iloc[iS]['reward']
            if iS==np.max(iSamples) or self.RL_data.iloc[iS+1].name==0:
                Sp  =   []
            else:
                Sp  =   self.RL_data.iloc[iS + 1]['state']
            # Do teaching
            agent.agent_move(S,A,R,Sp)

            count   +=  1
            if not count % 100:
                # Status bar
                stdout.write('\r')
                # the exact output you're looking for:
                stdout.write("Move %i/%i : [%-60s] %d%%, completed" % (count, len(iSamples), '=' * int(count / len(iSamples) * 60), 100 * count / len(iSamples)))
                stdout.flush()

                self.action_value   =   np.reshape( agent.action_value, [3, 5, 10, 10] )
                pickle.dump({'action_values':self.action_value}, open(path.join(repoSave, 'RL_action_values.p'), 'wb'))


    def make_line_dictionary(self):
        # Possible entries : [0,1,2]
        self.line_dictionary    =   {(0,0,0):0, (0,0,1):1, (0,1,1):2, (1,1,1):3,\
                                    (0,0,2):4, (0,2,2):5, (2,2,2):6, (0,1,2):7,\
                                    (1,1,2):8, (1,2,2):9}



class Season:

    def __init__(self, year):
        self.year   =   year


    def list_game_ids(self, repoPbP, repoPSt):
        # Format year
        iyear           =   self.year.replace('Season_', '')
        # Get data - long
        gc              =   Game(repoPbP, repoPSt, iyear)
        # Get game IDs
        self.games_id   =   gc.df.drop_duplicates(subset=['season', 'gcode'], keep='first')[['season', 'gcode', 'refdate', 'hometeam', 'awayteam']]


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
        self.lineShifts     =   self.lineShifts[self.lineShifts['equalstrength']]
        """dataFrame   =   self.lineShifts
        # Filter out powerplays
        isEqs       =   dataFrame['equalstrength']
        self.lineShifts =   {_key:np.array(dataFrame[_key])[np.array(isEqs)] for _key in dataFrame.keys()}"""


    def pick_regulartime(self):
        self.lineShifts     =   self.lineShifts[self.lineShifts['regulartime']]
        """dataFrame   =   self.lineShifts
        # Filter out overtime
        isRt        =   dataFrame['regulartime']
        self.lineShifts =   {_key:np.array(dataFrame[_key])[np.array(isRt)] for _key in dataFrame.keys()}"""


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


    def pull_line_shifts(self, team='home', minduration=None):
        # Pick the right team
        tmDict  =   {'home':'h', 'away':'a', 'both':'ha'}
        tmP     =   tmDict[team]

        # Make containers
        LINES       =   {'playersID':[], 'onice':[0], 'office':[], 'iceduration':[], 'SHOT':[0], 'GOAL':[0], 'BLOCK':[0], 'MISS':[0], 'PENL':[0], 'equalstrength':[True], 'regulartime':[], 'period':[], 'differential':[]}
        # Loop on all table entries
        prevDt      =   []
        prevLine    =   np.array([1, 1, 1])
        evTypes     =   ['GOAL', 'SHOT', 'PENL', 'BLOCK', 'MISS']
        if team=='both':
            prevLine=   (np.ones([1,3])[0], np.ones([1,3])[0])
        for idL, Line in self.df_wc.iterrows():
            if team=='both':
                curLine     =   ( np.sort(self.pull_offensive_players(Line, 'h')), np.sort(self.pull_offensive_players(Line, 'a')) )
                self.teams  =   [Line['hometeam'], Line['awayteam']]
            else:
                curLine     =   np.sort(self.pull_offensive_players(Line, tmP))
                self.teams  =   Line[team+'team']

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
                LINES['period'].append(prevDt['period'])
                LINES['regulartime'].append(prevDt['period'] < 4)
                LINES['differential'].append(np.sum(LINES['GOAL']))
                # Start new shift
                LINES['onice'].append(prevDt['seconds'])
                LINES['equalstrength'].append(prevDt['away.skaters']==6 and prevDt['home.skaters']==6)
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
        LINES['period'].append(prevDt['period'])
        LINES['regulartime'].append(prevDt['period']<4)
        LINES['differential'].append(np.sum(LINES['GOAL']))

        # Store
        self.lineShifts =   pd.DataFrame.from_dict(LINES)
        if not minduration is None:
            self.lineShifts     =   self.lineShifts[self.lineShifts['iceduration']>=minduration]


    def pull_players_classes(self, model, classifier, nGames=30):
        # List concerned players
        all_pl  =   self.lineShifts['playersID'].values
        if len(all_pl) == 0:
            self.player_classes = []
            return
        all_plC =   np.unique( np.concatenate(all_pl) )
        all_plN = self.rf.set_index('player.id').loc[all_plC[all_plC > 1]]['firstlast'].drop_duplicates(keep='first')
        if len(all_plN) == 0:
            self.player_classes = []
            return
        # Get players' team
        Hp      =   np.unique( np.concatenate([ x[0] for x in all_pl]) )
        Ap      =   np.unique( np.concatenate([x[1] for x in all_pl]) )
        #pTeam   =   [ np.where([x in Hp, x in Ap])[0] for x in all_plN.index.values]
        pTeam   =   [ self.teams[0] if x in Hp else self.teams[1] for x in all_plN.index.values]
        # Get raw player stats
        gcode   =   int( str(self.season)[:4]+'0'+str(self.gameId) )
        DT, dtCols  =   pull_stats(self.repoPSt, self.repoPbP, uptocode=gcode, nGames=nGames, plNames=all_plN.values)
        # --- Get player classes
        # pre-process data
        DT[dtCols]  =   ut_sanitize_matrix(DT[dtCols])
        DT_n, _     =   do_normalize_data(DT[dtCols], normalizer=model['normalizer'])
        DT_n_p, _   =   do_reduce_data(DT_n, pca=model['pca'])
        # model players' performance
        DTfeed      =   {classifier['annX']: DT_n_p}
        classif     =   classifier['sess'].run(classifier['annY'], feed_dict=DTfeed)
        # Get players class
        ctrLst      =   np.array( (model['global_centers']['selke'], model['global_centers']['ross'], model['global_centers']['poor']) )
        pl_class    =   [np.argmin( np.sum( (classif[x,:] - ctrLst)**2, axis=1 ) ) for x in range(classif.shape[0])]
        pl_class    =   pd.DataFrame(pl_class, columns=['class'], index=all_plN.index.values)
        pl_class.loc[:, 'firstlast']    =   all_plN
        pl_class.loc[:, 'pred_ross']    =   classif[:,0]
        pl_class.loc[:, 'pred_selke']   =   classif[:,1]
        pl_class.loc[:, 'team']         =   pTeam
        self.player_classes =   pl_class


    def recode_line(self, lineDict, line):
        if not type(line) is tuple:
            line    =   tuple(line)

        if line in lineDict.keys():
            return lineDict[line]
        else:
            return -1


    def encode_line_players(self):
        lComp   =   self.lineShifts['playersID']
        lineCode=   []
        for iR in lComp.index:
            row     =   lComp.loc[iR]
            nrow    =   []
            for iT in row:
                nTuple  =   []
                for iN in iT:
                    if iN in self.player_classes.index:
                        number  =   self.player_classes.loc[iN]['class']
                        if not type(number) is np.int64:
                            number  =   number.iloc[0]
                        nTuple.append( number )
                    else:
                        nTuple.append( -1 )
                nrow.append( tuple( np.sort(nTuple) ) )
            lineCode.append(nrow)
        return lineCode


    def recode_period(self, periods):
        return periods-1


    def recode_differential(self, differential):
        return np.minimum( np.maximum(differential, -2), 2 ) + 2


    def recode_reward(self, lineShift):
        return lineShift['GOAL'].values*5 + lineShift['SHOT'].values


    def recode_states(self, state1, state2, state3):
        return state3*50 + state2*10 + state1, 10*5*3


    def build_statespace(self, lineDict):
        # --- States
        # Encode line compositions
        lCode   =   self.encode_line_players()
        lComp   =   np.array( [[self.recode_line(lineDict, a) for a in b] for b in lCode] )
        # Remove -1
        remL    =   ~(lComp==-1).any(axis=1)
        lComp   =   lComp[remL,:]
        state1  =   lComp[:,0] # opposing line composition
        state2  =   self.recode_differential(self.lineShifts['differential'][remL].values)  # differential
        state3  =   self.recode_period( self.lineShifts['period'][remL].values )    # period
        state, nstates  =   self.recode_states( state1, state2, state3 )
        # Actions
        action, nactions=   lComp[:,1], len(lineDict)
        # Reward
        reward  =   self.recode_reward(self.lineShifts[remL])
        return state, action, reward, nstates, nactions, remL


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
repoCode    =   path.join(root, 'Code/Python')
repoModel   =   path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')
repoSave    =   None #path.join(repoCode, 'ReinforcementLearning/NHL/playbyplay/data')




# LEARN LINE VALUES
# =================
"""
HSS         =   HockeySS(repoPbP, repoPSt)
HSS.list_all_games()
HSS.pull_RL_data(repoModel, repoSave)
HSS.teach_RL_agent()



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

