from os import path

import numpy as np
import datetime
import collections
from typing import List, Tuple, Set

from ReinforcementLearning.NHL.playbyplay.players import players_classes
from ReinforcementLearning.NHL.playbyplay.season import Season
from ReinforcementLearning.NHL.playbyplay.shifts import LineShifts
from ReinforcementLearning.NHL.player.player_type import PlayerType
from ReinforcementLearning.NHL.playerstats.nhl_player_stats import PlayerStatsFetcher


class Game:

    def __init__(self, season: Season, gameId: int):
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
        self.home_team = self.df_wc["hometeam"].unique()[0]
        self.away_team = self.df_wc["awayteam"].unique()[0]
        self.teams_label_for_shift  =   "" # 'home', 'away' or 'both'
        #
        self.players_classes_cache = {} # cache for players' classes.
        self.players_classes_mgr = players_classes(game_data=self, model=self.season.preprocessing, classifier=self.season.classifier)

    def __str__(self):
        return "%s: game %d" % (self.season, self.gameId)

    def get_away_lines(self, accept_repeated=False) -> \
            List[Tuple[Tuple[int, int, int], Tuple[PlayerType, PlayerType, PlayerType], float]]:
        """
        
        Args:
            accept_repeated: if true, lines can have repeating players.

        Returns: A sorted list where each element contains:
        * the line as id's
        * the line as categories
        * the number of seconds played. The higher the number of seconds played the higher this element is.

        """


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
        if accept_repeated:
            # result_no_classes = dict(sorted_tuples_and_secs[:4])
            result_no_classes = sorted_tuples_and_secs[:4]
        else:
            players_used = set()
            lines_chosen = []
            for line, secs in sorted_tuples_and_secs:
                # is this line using players already present?
                line_as_set = set(line)
                if (not 1 in line_as_set) and (accept_repeated or (len(players_used.intersection(line_as_set)) == 0)):
                    lines_chosen.append((line, a_dict[line]))
                    players_used = players_used.union(line_as_set)
                    if len(lines_chosen) == 4:
                        break # horrible, but effective
            # result_no_classes = dict(lines_chosen[:4])
            result_no_classes = lines_chosen[:4]
        result_with_classes = list(map(
            # Here the sort is to make sure that the conversion from player categories to line categories can be done
            # without enumerating all possible combinations of the same players in the dictionary
            # for example, lines [2,0,1], [1,2,0], [0,2,1] and [0,1,2] can be represented by a single line category in the dict
            lambda line_with_secs: (line_with_secs[0], tuple(np.sort(self.classes_of_line(line_with_secs[0]))), line_with_secs[1]),
            result_no_classes))
        return result_with_classes

    def classes_of_line(self, a: List[int]) -> List[PlayerType]:
        """Returns classes of members of a line given their id's."""
        player_classes = self.players_classes_mgr.get(equal_strength=True, regular_time=True, min_duration=20, nGames=30)
        return list(map(PlayerType.from_int, player_classes.loc[list(a)]["class"].values))

    def __get_players_from__(self, team_name: str) -> Set[int]:
        players_classes = self.players_classes_mgr.get(True, True, 20, nGames=30) # TODO: why this specific call?
        return set(players_classes[players_classes["team"] == team_name].index)

    def get_ids_of_home_players(self) -> Set[int]:
        return self.__get_players_from__(team_name=self.home_team)

    def get_ids_of_away_players(self) -> Set[int]:
        return self.__get_players_from__(team_name=self.away_team)

    def player_id_to_name(self, player_id: int) -> str:
        return self.rf_wc[self.rf_wc['player.id'] == player_id]['numfirstlast'].tolist()[0]

    def formation_ids_to_names(self, formation: List[List[int]]) -> List[List[str]]:
        ids = self.get_ids_of_home_players()
        # let's translate these numbers into names:
        # input is ~ [(656, 27, 31), (1380, 389, 1035), (8, 9, 1164), (281, 13, 14)]
        return list(map(lambda a_line: list(map(self.player_id_to_name, a_line)), formation))

    def formation_ids_to_str(self, formation: List[List[int]]) -> str:
        result = []
        lines_with_names = self.formation_ids_to_names(formation)
        line_no = 1
        for a_line in lines_with_names:
            first_guy, second_guy, third_guy = a_line
            result.append("Line %d: %s, %s, %s" % (line_no, first_guy, second_guy, third_guy))
            line_no += 1
        return '\n'.join(result)

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

Formation = collections.namedtuple('Formation', 'as_names as_categories')

def get_lines_for(season: Season, base_date: datetime.date, how_many_days_back: int, team_abbrev: str) -> Formation:
    """prediction of the lines that the 'away' team will use."""
    assert(how_many_days_back >= 0)
    ids = season.get_last_n_away_games_since(base_date, n=how_many_days_back, team_abbrev=team_abbrev)
    lines_dict = {}
    for game_id in ids:
        g = Game(season, gameId=game_id)
        print("Processing game %s" % (g))
        result_as_list = g.get_away_lines()
        for line_as_ids, line_as_types, secs_played in result_as_list:
            line_as_ids = tuple(map(g.player_id_to_name, line_as_ids))
            if line_as_ids in lines_dict:
                # update number of seconds played
                lines_dict[line_as_ids] = (line_as_types, lines_dict[line_as_ids][1] + secs_played)
            else:
                # seed entry in dictionary
                lines_dict[line_as_ids] = (line_as_types, secs_played)
    print("DONE")

    # for k, v in lines_dict.items():
    #     print(k, v)
    # ok, now sort by seconds played, keep top 4:
    flat_list = list(map(lambda x: (x[0], x[1][0], x[1][1]), lines_dict.items()))
    result_as_list = sorted(flat_list, key=lambda x: x[2], reverse=True)
    print("%d lines used consistently" % (len(result_as_list)))
    for a_line, a_cat, num_secs in result_as_list:
        print("%s played %.2f secs" % (a_line, num_secs))
    top_4_as_list = result_as_list[:4]
    print("Keeping top 4:")
    for a_line, a_cat, num_secs in top_4_as_list:
        print("%s played %.2f secs" % (a_line, num_secs))
    away_lines_names = list(map(lambda x: x[0], top_4_as_list))  # as names
    # print(away_lines_names)
    away_lines = list(map(lambda x: x[1], top_4_as_list))  # as categories
    # print(away_lines)
    return Formation(as_names=away_lines_names, as_categories=away_lines)


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


