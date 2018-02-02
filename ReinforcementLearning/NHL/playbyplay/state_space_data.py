import pickle
from os import path
from random import shuffle
from sys import stdout

import numpy as np
import pandas as pd
import tensorflow as tf

from ReinforcementLearning.NHL.playerstats.ann_classifier import *
from ReinforcementLearning.NHL.playbyplay.agent import Agent
from ReinforcementLearning.NHL.playbyplay.playbyplay_data import Season
from Utils.programming.ut_find_folders import ut_find_folders


class HockeySS:

    def __init__(self, db_root):
        self.db_root    =   db_root
        self.repoPbP    =   path.join(db_root, 'PlayByPlay')
        self.repoPSt    =   path.join(db_root, "PlayerStats", "player")
        self.seasons    =   ut_find_folders(self.repoPbP, True)


    def list_all_games(self):
        # List games
        games_lst       =   pd.DataFrame()
        for iy in self.seasons:
            iSea        =   Season( self.db_root, int(iy.replace('Season_', '')[:4]) )
            games_lst   =   pd.concat( (games_lst, iSea.games_id), axis=0 )
        self.games_lst  =   games_lst


    def pull_RL_data(self, repoModel, repoSave=None):
        # Prepare players model: reload info
        CLS                 =   ANN_classifier()
        sess, annX, annY    =   CLS.ann_reload_model(repoModel)
        self.classifier     =   {'annX':annX, 'annY':annY, 'sess':sess}
        self.players_model  =   pickle.load(open(path.join(repoModel, 'baseVariables.p'), 'rb'))
        # Make lines dictionary
        self.make_line_dictionary()
        # List line shifts
        RL_data     =   pd.DataFrame()
        GAME_data   =   pd.DataFrame()
        PLAYER_data =   pd.DataFrame()
        count       =   0
        allR        =   []

        # Loop on seasons
        for iy in np.unique(self.games_lst['season'].values):

            # Extract season data
            iSea    =   Season( self.db_root, int( str(iy)[:4]) )

            # List games
            games   =   self.games_lst[self.games_lst['season']==iy]

            # Loop on games
            for ic, ih, ia in zip(games['gcode'].values, games['hometeam'].values, games['awayteam'].values):

                # Extract game data
                iGame   =   iSea.pick_game(ic)

                # Check if some data was retrieved:
                if len(iGame.df_wc)>0:
                    iGame.pull_line_shifts('both', minduration=20)
                    iGame.pick_regulartime()
                    iGame.pick_equalstrength()
                    iGame.pull_players_classes(self.players_model, {'annX':annX, 'annY':annY, 'sess':sess})
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



# ========
# LAUNCHER

# Pointers
db_root     =   '/home/younesz/Documents/Databases/Hockey'
#root        =   '/Users/younes_zerouali/Documents/Stradigi'
repoPbP     =   path.join(db_root, 'Databases/Hockey/PlayByPlay')
repoPSt     =   path.join(db_root, 'Databases/Hockey/PlayerStats/player')
repoCode    =   path.join('/home/younesz/Documents/Code/NHL_stats_SL')
repoModel   =   path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')
repoSave    =   None #path.join(repoCode, 'ReinforcementLearning/NHL/playbyplay/data')

"""
# Execute functions
HSS         =   HockeySS(db_root)
HSS.list_all_games()
HSS.pull_RL_data(repoModel, repoSave)
HSS.teach_RL_agent()

"""
