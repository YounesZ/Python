import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
from typing import List, Tuple, Set
from os import path

from ReinforcementLearning.NHL.playerstats.nhl_player_stats import PlayerStatsFetcher
from ReinforcementLearning.NHL.player.player_type import PlayerType
from ReinforcementLearning.NHL.playbyplay.players import players_classes

class Season:
    """Encapsulates all elements for a season."""

    def __init__(self, db_root: str, repo_model: str, year_begin: int):
        self.db_root    =   db_root
        self.repo_model = repo_model
        self.year_begin =   year_begin
        self.year_end   =   self.year_begin + 1

        # List games and load season data
        self.list_game_ids()

    def list_game_ids(self):
        self.repoPbP    =   path.join(self.db_root, 'PlayByPlay')
        self.repoPSt    =   path.join(self.db_root, "PlayerStats", "player")
        # Get data - long
        self.load_data()
        # Get game IDs
        self.games_id   =   self.dataFrames['playbyplay'].drop_duplicates(subset=['season', 'gcode'], keep='first')[['season', 'gcode', 'refdate', 'hometeam', 'awayteam']]

    def load_data(self):
        dataPath        =   path.join(self.repoPbP, 'Season_%d%d' % (self.year_begin, self.year_end),'converted_data.p')
        self.dataFrames =   pickle.load(open(dataPath, 'rb'))

    def pick_game(self, gameId):
        return Game(self, gameId, self.repo_model)

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
            gameInfo    =   pickle.load( open(path.join(db_root, 'processed', 'gamesInfo.p'), 'rb') )
            gameInfo    =   gameInfo[gameInfo['gameDate']==date_as_str][gameInfo['teamAbbrev']==home_team_abbr]
            gameId      =   gameInfo['gameId']
            gameId      =   int( gameId.values.astype('str')[0][5:] )
            return gameId
        except Exception as e:
            raise IndexError("There was no game for '%s' on '%s'" % (home_team_abbr, date_as_str))

class LineShifts(object):
    """Encapsulates queries done to determine line shifts."""

    def __init__(self, game):
        self.__lineShifts   =   None # TODO: call this 'data'
        self.equal_strength =   True
        self.regular_time   =   True
        self.min_duration   =   0 # minimum number of seconds for which we want to consider shifts.
        self.team   =   'both' # 'home', 'away' or 'both'
        # Pick the right team
        team        =   'both'
        tmP         =   {'home': 'h', 'away': 'a', 'both': 'ha'}[team]

        # Make containers
        LINES = {
            'playersID': [],
            'home_line': [],
            'away_line': [],
            'onice': [0],
            'office': [],
            'iceduration': [],
            'SHOT': [0],
            'GOAL': [0],
            'BLOCK': [0],
            'MISS': [0],
            'PENL': [0],
            'equalstrength': [True],
            'regulartime': [],
            'period': [],
            'differential': []
        }
        # Loop on all table entries
        prevDt          =   []
        prev_home_line  =   prev_away_line = np.ones([1, 3])[0]
        # prevLine = (np.ones([1, 3])[0], np.ones([1, 3])[0]) if team == 'both' else np.array([1, 1, 1])
        evTypes         =   ['GOAL', 'SHOT', 'PENL', 'BLOCK', 'MISS']
        for idL, Line in game.df_wc.iterrows():
            home_line   =   np.sort(game.pull_offensive_players(Line, 'h'))
            away_line   =   np.sort(game.pull_offensive_players(Line, 'a'))
            self.teams  =   [Line['hometeam'], Line['awayteam']]
            # curLine = (home_line, away_line)
            # if team == 'both':
            #     curLine = (home_line, away_line)
            #     teams = [Line['hometeam'], Line['awayteam']]
            # else:
            #     curLine = np.sort(game.pull_offensive_players(Line, tmP))
            #     teams = Line[team + 'team']

            # team of interest has changed?
            if len(prevDt) == 0:
                prevDt  =   Line
                thch    =   False
            else:
                thch    =   not (prev_home_line == home_line).all() or not (prev_away_line == away_line).all()
            # elif team == 'both':
            #     thch = not (prevLine[0] == curLine[0]).all() or not (prevLine[1] == curLine[1]).all()
            # else:
            #     thch = not (prevLine == curLine).all()

            if thch:
                # Terminate this shift
                LINES['playersID'].append((prev_home_line, prev_away_line))
                LINES['home_line'].append(prev_home_line)
                LINES['away_line'].append(prev_away_line)
                LINES['office'].append(prevDt['seconds'])
                LINES['iceduration'].append(LINES['office'][-1] - LINES['onice'][-1])
                LINES['period'].append(prevDt['period'])
                LINES['regulartime'].append(prevDt['period'] < 4)
                LINES['differential'].append(np.sum(LINES['GOAL']))
                # Start new shift
                LINES['onice'].append(prevDt['seconds'])
                LINES['equalstrength'].append(prevDt['away.skaters'] == 6 and prevDt['home.skaters'] == 6)
                LINES['SHOT'].append(0)
                LINES['GOAL'].append(0)
                LINES['PENL'].append(0)
                LINES['BLOCK'].append(0)
                LINES['MISS'].append(0)
            if any([x == Line['etype'] for x in evTypes]):
                sign    =   int(Line['hometeam'] == Line['ev.team']) * 2 - 1
                LINES[Line['etype']][-1] += sign
                if Line['etype'] == 'GOAL':
                    LINES['SHOT'][-1] += sign
            if Line['etype'] == 'PENL':
                LINES['equalstrength'][-1] = False
            prevDt      =   deepcopy(Line)
            prev_home_line = deepcopy(home_line)
            prev_away_line = deepcopy(away_line)
            # prevLine = deepcopy(curLine)

        # Terminate line history
        LINES['office'].append(Line['seconds'])
        LINES['iceduration'].append(LINES['office'][-1] - LINES['onice'][-1])
        LINES['playersID'].append((prev_home_line, prev_away_line))
        LINES['home_line'].append(prev_home_line)
        LINES['away_line'].append(prev_away_line)
        LINES['period'].append(prevDt['period'])
        LINES['regulartime'].append(prevDt['period'] < 4)
        LINES['differential'].append(np.sum(LINES['GOAL']))

        # ok, now let's buid it:
        self.__lineShifts = pd.DataFrame.from_dict(LINES)
        # # all done, then:
        # return (team, teams, lineShifts)

    def as_df(self, team: str, equal_strength: bool, regular_time: bool, min_duration: int) -> pd.DataFrame:
        """Gets line shifts as a data frame."""
        df = self.__lineShifts
        if equal_strength:
            df = df[df['equalstrength']]
        if regular_time:
            df = df[df['regulartime']]
        if not min_duration is None:
            df = df[df['iceduration'] >= min_duration]
        # for which team(s).
        if team == 'both':
            pass
            #print(df.columns.names)
            # df = df.drop(columns=['home_line', 'away_line']) # TODO: see https://github.com/pandas-dev/pandas/issues/19078
        elif team == 'home':
            # df = df.drop(columns=['playersID']) # TODO: see https://github.com/pandas-dev/pandas/issues/19078
            df = df.drop(['playersID'], axis=1)
            df = df.rename(columns={'home_line': 'playersID'})
        elif team == 'away':
            # df = df.drop(columns=['playersID']) # TODO: see https://github.com/pandas-dev/pandas/issues/19078
            df = df.drop(['playersID'], axis=1)
            df = df.rename(columns={'away_line': 'playersID'})
        else:
            raise RuntimeError("Can't choose elements from team '%s'" % (team))
        return df

    def __update__(self):
        pass

class Game:

    def __init__(self, season: Season, gameId: int, repo_model: str):
        # Retrieve game info
        self.season =   season
        self.gameId =   gameId

        # Get all player names
        self.df     =   season.dataFrames['playbyplay']
        self.df_wc  =   self.df[self.df['gcode'] == gameId]
        self.hd     =   self.df_wc.columns

        # let's keep the roster only for players that we are interested in:
        fields_with_ids =   ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'away.G', 'home.G']
        all_sets        =   list(map(lambda field_id: set(self.df_wc[field_id].unique().tolist()).difference({1}),  # '1' is not a real id.
                            fields_with_ids))
        all_ids_of_players= set.union(*all_sets)
        self.rf         =   season.dataFrames['roster']
        self.rf_wc      =   self.rf[self.rf['player.id'].isin(all_ids_of_players)]

        # Fetch line shifts
        self.player_classes     = 	None # structure containing all player's classes (categories).
        self.stats_fetcher      =   PlayerStatsFetcher(repoPSt=season.repoPSt, repoPbP=season.repoPbP, do_data_cache=True)
        # let's keep the roster only for players that we are interested in:
        # Fetch line shifts
        self.shifts_equal_strength  =   True
        self.shifts_regular_time    =   True
        self.lineShifts             =   LineShifts(self)
        self.teams                  =   None
        self.teams_label_for_shift  =   "" # 'home', 'away' or 'both'
        #
        self.players_classes_cache = {} # cache for players' classes.
        self.players_classes_mgr = players_classes.from_repo(game_data=self, repoModel=repo_model)

    # This is deprecated: this functionality is now Season()'s job
    """
    def get_game_ids(self):
        "List all game numbers"
        return np.unique(self.df['gcode'])
    """

    def get_away_lines(self, accept_repeated=False) -> Tuple[pd.DataFrame, List[List[PlayerType]]]:
        """
        Calculates top lines used by opposing team. 
        Each line returned contains the CATEGORY of each player.
        """
        lineShifts = self.lineShifts.as_df(team='away', equal_strength=self.shifts_equal_strength, regular_time=self.shifts_regular_time, min_duration=20)
        # teams_label_for_shift, teams, lineShifts = self.calculate_line_shifts(team='away', minduration=20)
        df = lineShifts.\
            groupby(by=lineShifts['playersID'].apply(tuple)).\
            agg({'iceduration': sum}).\
            sort_values(by=['iceduration'], ascending=False)

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
        # Here the sort is to make sure that the conversion from player categories to line categories can be done
        # without enumerating all possible combinations of the same players in the dictionary
        # for example, lines [2,0,1], [1,2,0], [0,2,1] and [0,1,2] can be represented by a single line category in the dict
        return (df, list(map(list, [np.sort(self.classes_of_line(a)) for a in lines_chosen])))

    def classes_of_line(self, a: List[int]) -> List[PlayerType]:
        """Returns classes of members of a line given their id's."""
        player_classes = self.players_classes_mgr.get(equal_strength=True, regular_time=True, min_duration=20, nGames=30)
        return list(map(PlayerType.from_int, player_classes.loc[list(a)]["class"].values))

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

    def __get_players_from__(self, repoModel:str, team_name: str) -> Set[int]:
        players_classes = self.players_classes_mgr.get(True, True, 20, nGames=30)
        return set(players_classes[players_classes["team"] == team_name].index)

    def get_home_players(self, repoModel:str) -> Set[int]:
        return self.__get_players_from__(repoModel, team_name=self.df_wc['hometeam'].iloc[0])

    def get_away_players(self, repoModel:str) -> Set[int]:
        return self.__get_players_from__(repoModel, team_name=self.df_wc['awayteam'].iloc[0])

    def pull_offensive_players(self, dfRow, team='h'):
        # Get player IDs
        pID     =   set([dfRow[team+str(x)] for x in range(1,7)])
        pID.discard(1) # '1' is not a true player ID.
        pID     =   list(pID)
        # Check positions
        pPOS    =   [self.rf_wc.loc[self.rf_wc['player.id']==x, 'pos'] for x in pID]
        pOFF    =   [(x.values[0]=='R' or x.values[0]=='L' or x.values[0]=='C') for x in pPOS]
        result = (list( np.array(pID)[pOFF] )+[1,1,1])[:3]
        return result

    def recode_line(self, lineDict, line):
        if not type(line) is tuple:
            line    =   tuple(line)

        if line in lineDict.keys():
            return lineDict[line]
        else:
            return -1


    def encode_line_players(self):
        players_classes = self.players_classes_mgr.get(equal_strength=True, regular_time=True, min_duration=20, nGames=30) # TODO: why these parameters??
        lineShifts  =   self.lineShifts.as_df(team='both', equal_strength=self.shifts_equal_strength, regular_time=self.shifts_regular_time, min_duration=20)
        lComp       =   lineShifts['playersID']
        lineCode    =   []
        for iR in lComp.index:
            row     =   lComp.loc[iR]
            nrow    =   []
            for iT in row:
                nTuple  =   []
                for iN in iT:
                    if iN in players_classes.index:
                        number  =   players_classes.loc[iN]['class']
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
        lCode           =   self.encode_line_players()
        lComp           =   np.array( [[self.recode_line(lineDict, a) for a in b] for b in lCode] )
        # Remove -1
        remL            =   ~(lComp==-1).any(axis=1)
        lComp           =   lComp[remL,:]
        state1          =   lComp[:,0] # opposing line composition
        lineShifts      =   self.lineShifts.as_df(team='both', equal_strength=self.shifts_equal_strength, regular_time=self.shifts_regular_time, min_duration=20)
        state2          =   self.recode_differential(lineShifts['differential'][remL].values)  # differential
        state3          =   self.recode_period(lineShifts['period'][remL].values )    # period
        state, nstates  =   self.recode_states( state1, state2, state3 )
        # Actions
        action, nactions=   lComp[:,1], len(lineDict)
        # Reward
        reward          =   self.recode_reward(lineShifts[remL])
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
repoSave    =   None #path.join(repoCode, 'ReinforcementLearning/NHL/playbyplay/data')




# LEARN LINE VALUES
# =================
"""

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


