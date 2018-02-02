import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple
from os import path
from ReinforcementLearning.NHL.playerstats.nhl_player_stats import pull_stats, do_normalize_data, do_reduce_data
from Utils.programming.ut_sanitize_matrix import ut_sanitize_matrix


class Season:
    """Encapsualtes all elements for a season."""

    def __init__(self, db_root: str, year_begin: int):
        self.db_root    =   db_root
        self.year_begin =   year_begin
        self.year_end   =   self.year_begin + 1

        # List games and load season data
        self.list_game_ids()
    #  def __init__(self, year_encoding):
    #     self.year_encoding   =   year_encoding # eg, 'Season_20122013'

    def list_game_ids(self):
        self.repoPbP    =   path.join(self.db_root, 'PlayByPlay')
        self.repoPSt    =   path.join(self.db_root, "PlayerStats", "player")
        # Get data - long
        self.games_data =   self.load_data()
        # Get game IDs
        self.games_id   =   self.game_active.df.drop_duplicates(subset=['season', 'gcode'], keep='first')[['season', 'gcode', 'refdate', 'hometeam', 'awayteam']]

    def load_data(self):
        dataPath        =   path.join(self.repoPbP, 'Season_%d%d' % (self.year_begin, self.year_end),'converted_data.p')
        self.dataFrames =   pickle.load(open(dataPath, 'rb'))

    def pick_game(self, gameId):
        return Game(self, gameId)

    def __str__(self):
        return "Season %d-%d" % (self.year_begin, self.year_end)


    @classmethod
    def get_game_id(cls, db_root: str, home_team_abbr: str, date_as_str: str) -> int:
        """
        let's convert game date to game code.
        For example Montreal received Ottawa on march 13, 2013 =>
            gameId = get_game_id(home_team_abbr='MTL', date_as_str='2013-03-13')
        """
        # TODO: make it fit with the class signature. For now it's pretty much standalone.
        try:
            gameInfo    =   pickle.load( open(path.join(db_root, 'gamesInfo.p'), 'rb') )
            gameInfo    =   gameInfo[gameInfo['gameDate']==date_as_str][gameInfo['teamAbbrev']==home_team_abbr]
            gameId      =   gameInfo['gameId']
            gameId      =   int( gameId.values.astype('str')[0][5:] )
            return gameId
        except Exception as e:
            raise IndexError("There was no game for '%s' on '%s'" % (home_team_abbr, date_as_str))


class Game:

    def __init__(self, season: Season, gameId: int):
        # Retrieve game info
        self.season =   season
        self.gameId =   gameId

        # Get all player names
        self.df_wc  =   season.dataFrames['playbyplay'][season.dataFrames['playbyplay']['gcode'] == gameId]
        self.hd     =   self.df_wc.columns

        # let's keep the roster only for players that we are interested in:
        fields_with_ids =   ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'away.G', 'home.G']
        all_sets        =   list(map(lambda field_id: set(self.df_wc[field_id].unique().tolist()).difference({1}),  # '1' is not a real id.
                            fields_with_ids))
        all_ids_of_players= set.union(*all_sets)
        self.rf_wc      =   season.dataFrames['roster'][season.dataFrames['roster']['player.id'].isin(all_ids_of_players)]

        # Fetch line shifts
        self.player_classes     = 	None # structure containing all player's classes (categories).
        self.lineShifts         =   {}
        self.teams              =   None
        self.teams_label_for_shift = "" # 'home', 'away' or 'both'

    # This is deprecated: this functionality is now Season()'s job
    """
    def get_game_ids(self):
        "List all game numbers"
        return np.unique(self.df['gcode'])
    """

    def get_away_lines(self, accept_repeated=False) -> Tuple[pd.DataFrame, List[List[int]]]:
        """
        Calculates top lines used by opposing team. 
        Each line returned contains the CATEGORY of each player.
        """
        teams_label_for_shift, teams, lineShifts = self.calculate_line_shifts(team='away', minduration=20)
        df = lineShifts.groupby(by=lineShifts['playersID'].apply(tuple)).agg(
            {'iceduration': sum}).sort_values(by=['iceduration'], ascending=False)

        a_dict = df.to_dict()['iceduration']
        sorted_tuples_and_secs = sorted(a_dict.items(), key=lambda x: x[1], reverse=True)
        players_used = set()
        lines_chosen = []
        for line, secs in sorted_tuples_and_secs:
            # is this line using players already present?
            line_as_set = set(line)
            if (not 1 in line_as_set) and (accept_repeated or (len(players_used.intersection(line_as_set)) == 0)):
                lines_chosen.append(line)
                players_used = players_used.union(line_as_set)
                if len(lines_chosen) == 4:
                    break # horrible, but effective
        return (df, list(map(list, [np.sort(self.classes_of_line(a)) for a in lines_chosen]))) # TODO: not sure about the 'sort'. What is it for?


    def classes_of_line(self, a: List[int]) -> List[int]:
        """Returns classes of members of a line given their id's."""
        return self.player_classes.loc[list(a)]["class"].values


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
        pID     =   set([dfRow[team+str(x)] for x in range(1,7)])
        pID.discard(1) # '1' is not a true player ID.
        pID     =   list(pID)
        # Check positions
        pPOS    =   [self.rf_wc.loc[self.rf_wc['player.id']==x, 'pos'] for x in pID]
        pOFF    =   [(x.values[0]=='R' or x.values[0]=='L' or x.values[0]=='C') for x in pPOS]
        return (list( np.array(pID)[pOFF] )+[1,1,1])[:3]

    def calculate_line_shifts(self, team='home', minduration: int=20): # ->Tuple[ pd.DataFrame:
        """
        Calculates the line shifts and returns them on a proper structure.
        This calculation is purely functional: it does not change the state of this class.
        Args:
            team: 'home', 'away', or 'both'
            minduration: time in seconds.

        Returns:

        """

        # Pick the right team
        tmDict  =   {'home':'h', 'away':'a', 'both':'ha'}
        tmP     =   tmDict[team]

        # Make containers
        LINES       =   {
            'playersID':[],
            'onice':[0],
            'office':[],
            'iceduration':[],
            'SHOT':[0],
            'GOAL':[0],
            'BLOCK':[0],
            'MISS':[0],
            'PENL':[0],
            'equalstrength':[True],
            'regulartime':[],
            'period':[],
            'differential':[]
        }
        # Loop on all table entries
        prevDt      =   []
        prevLine    =   np.array([1, 1, 1])
        evTypes     =   ['GOAL', 'SHOT', 'PENL', 'BLOCK', 'MISS']
        if team=='both':
            prevLine=   (np.ones([1,3])[0], np.ones([1,3])[0])
        for idL, Line in self.df_wc.iterrows():
            if team=='both':
                curLine     =   ( np.sort(self.pull_offensive_players(Line, 'h')), np.sort(self.pull_offensive_players(Line, 'a')) )
                teams  =   [Line['hometeam'], Line['awayteam']]
            else:
                curLine     =   np.sort(self.pull_offensive_players(Line, tmP))
                teams  =   Line[team+'team']

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

        # ok, now let's buid it:
        lineShifts =   pd.DataFrame.from_dict(LINES)
        if not minduration is None:
            lineShifts     =   lineShifts[lineShifts['iceduration']>=minduration]
        # all done, then:
        return (team, teams, lineShifts)


    def pull_line_shifts(self, team: str='home', minduration: int=20):
        self.teams_label_for_shift, self.teams, self.lineShifts = self.calculate_line_shifts(team, minduration)

    def pull_players_classes_from_repo_address(self, repoModel:str, number_of_games=30):
        """
        Calculates (dataframe with) all player's classes.
        Updates the 'data for game' structure with it; also returns it.
        Args:
            repoModel: folder where the model is saved.
            number_of_games: number of games we want to analyze.

        Returns:
            Player classes
            
        Examples:
            repoModel = ... # here goes the directory where your model is saved.
            # Montreal received Ottawa on march 13, 2013
            gameId = Season.get_game_id(home_team_abbr='MTL', date_as_str='2013-03-13')
            season      =   '20122013'
            mtlott      =   Game(repoPbP, repoPSt, season, gameId=gameId )
            #
            players_classes = get_players_classes(repoModel, mtlott, number_of_games=30)
            # this is equivalent to ask 'mtlott' for the data; so:
            assert players_classes.equals(mtlott.player_classes)
        """
        # Need to load the data pre-processing variables
        preprocessing = pickle.load(open(path.join(repoModel, 'baseVariables.p'), 'rb'))

        # Need to load the classification model (for players' predicted ranking on trophies voting lists)
        classifier = {'sess': tf.Session(), 'annX': [], 'annY': []}
        saver = tf.train.import_meta_graph(path.join(repoModel, path.basename(repoModel) + '.meta'))
        graph = classifier['sess'].graph
        classifier['annX'] = graph.get_tensor_by_name('Input_to_the_network-player_features:0')
        classifier['annY'] = graph.get_tensor_by_name('prediction:0')
        saver.restore(classifier['sess'], tf.train.latest_checkpoint(path.join(repoModel, './')))

        self.pull_players_classes(preprocessing, classifier, nGames=number_of_games)
        return self.player_classes

    def pull_players_classes(self, model, classifier, nGames=30):
        """
        Gets classes players represented in the shifts for this game.
        Assumes that these shifts are specified (by a previous call to 'pull_line_shifts'
        If the shifts are not specified throws an error.
        Args:
            model: 
            classifier: 
            nGames: 

        Returns:
            Player classes
            

        """

        # List concerned players
        teams_label_for_shift, teams, all_line_shifts = self.calculate_line_shifts(team='both')
        all_pl  =   all_line_shifts['playersID'].values
        if len(all_pl) == 0:
            self.player_classes = []
            return
        all_plC =   np.unique( np.concatenate(all_pl) )
        all_plN = self.rf_wc.set_index('player.id').loc[all_plC[all_plC > 1]]['firstlast'].drop_duplicates(keep='first')
        if len(all_plN) == 0:
            self.player_classes = []
            return
        # Get players' team
        Hp      =   np.unique( np.concatenate([ x[0] for x in all_pl]) )
        Ap      =   np.unique( np.concatenate([x[1] for x in all_pl]) )
        #pTeam   =   [ np.where([x in Hp, x in Ap])[0] for x in all_plN.index.values]
        pTeam   =   [ teams[0] if x in Hp else teams[1] for x in all_plN.index.values]
        # Get raw player stats
        # gcode   =   int( str(self.season)[:4]+'0'+str(self.gameId) )
        gcode   =   int( str(self.season.year_begin)+'0'+str(self.gameId) )
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
        assert self.teams_label_for_shift == 'both', \
            "Encoding lines only possible when I have shift information of both teams (currently: '%s')." % (self.teams_label_for_shift)
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
repoCode    =   path.join(root, 'Code/NHL_stats_SL')
repoModel   =   path.join(repoCode, 'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')
repoSave    =   path.join(repoCode, 'ReinforcementLearning/NHL/playbyplay/data')




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

