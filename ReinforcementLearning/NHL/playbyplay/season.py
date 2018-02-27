import pickle
import datetime
import logging
import collections
from os import path
from typing import Tuple, Optional, Set

from ReinforcementLearning.NHL.playbyplay.players import get_model_and_classifier_from

Formation = collections.namedtuple('Formation', 'as_names as_categories')

class Season:
    """Encapsulates all elements for a season."""

    def __init__(self, alogger: logging.Logger, db_root: str, repo_model: str, year_begin: int):
        self.alogger = alogger
        self.db_root    =   db_root
        self.repo_model = repo_model
        self.year_begin =   year_begin
        self.year_end   =   self.year_begin + 1

        # Need to load the data pre-processing variables
        self.preprocessing, self.classifier = get_model_and_classifier_from(self.repo_model)


        # List games and load season data
        self.repoPbP    =   path.join(self.db_root, 'PlayByPlay')
        self.repoPSt    =   path.join(self.db_root, "PlayerStats", "player")
        # Get data - long
        dataPath        =   path.join(self.repoPbP, 'Season_%d%d' % (self.year_begin, self.year_end),'converted_data.p')
        self.dataFrames =   pickle.load(open(dataPath, 'rb'))
        # Get game IDs
        self.games_id   =   self.dataFrames['playbyplay'].\
            drop_duplicates(subset=['season', 'gcode'], keep='first')[['season', 'gcode', 'refdate', 'hometeam', 'awayteam']]
        #
        self.games_info = pickle.load(open(path.join(self.db_root, 'processed', 'gamesInfo.p'), 'rb'))
        self.games_info = self.games_info[
            (self.games_info['gameDate'] >= ('%d-09-01' % (self.year_begin))) & (self.games_info['gameDate'] <= ('%d-07-01' % (self.year_end)))] # take games only for this season
        self.games_info = self.games_info.sort_values(by=['gameDate'], ascending=False)
        self.away_lines_per_game = {}

    def __strip_game_id__(self, game_id_as_str: str) -> str:
        "A game id of 23456 for year 2012 will be shown as '2012023456'. This function strips it."
        return game_id_as_str[5:]

    def get_teams(self) -> Set[str]:
        """Teams that played in this season."""
        return set(self.games_id.hometeam.unique())

    def get_game_at_or_just_before(self, game_date: datetime.date, home_team_abbr: str, delta_in_days: int = 3) -> Optional[Tuple[int, datetime.date]]:
        """
        let's convert game date to game code.
        For example Montreal received Ottawa on march 13, 2013 =>
            gameId = get_game_id(home_team_abbr='MTL', date_as_str=datetime.date(year=2013, month=3, day=13))
        """
        delta_in_days = datetime.timedelta(days=delta_in_days)
        earliest_date = game_date - delta_in_days
        date_as_str = str(game_date)
        earliest_date_as_str = str(earliest_date)
        try:
            # gameInfo = self.games_info[
            #     (self.games_info['gameDate'] <= date_as_str) & (self.games_info['gameDate'] >= earliest_date_as_str)
            # ][self.games_info['teamAbbrev'] == home_team_abbr]
            gameInfo = self.games_info[
                (self.games_info['gameDate'] <= date_as_str) &
                (self.games_info['gameDate'] >= earliest_date_as_str) &
                (self.games_info['teamAbbrev'] == home_team_abbr)]
            # I should have 1 id per row, without repetitions:
            assert len(gameInfo["gameId"].unique()) == len(gameInfo.index)
            gameInfo = gameInfo.head(1)
            gameId = gameInfo['gameId']
            top_game_date_as_str = gameInfo['gameDate'].values.astype('str')[0]
            top_game_date=datetime.datetime.strptime(top_game_date_as_str, "%Y-%m-%d").date()
            gameId = int(gameId.values.astype('str')[0][5:])
            return (gameId, top_game_date)
        except Exception as e:
            # TODO: write in log e.get_message()
            return None

    def get_game_id(self, home_team_abbr: str, game_date: datetime.date) -> int:
        """
        let's convert game date to game code.
        For example Montreal received Ottawa on march 13, 2013 =>
            gameId = get_game_id(home_team_abbr='MTL', date_as_str=datetime.date(year=2013, month=3, day=13))
        """
        result = self.get_game_at_or_just_before(game_date, home_team_abbr, delta_in_days = 0)
        if result is None:
            raise IndexError("There was no game for '%s' on '%s'" % (home_team_abbr, str(game_date)))
        gameId, _ = result
        return gameId

    def get_last_n_home_games_for(self, n: int, team_abbrev: str) -> Set[int]:
        raise NotImplementedError("lala")

    def get_last_n_away_games_since(self, a_date: datetime.date, n: int, team_abbrev: str) -> Set[int]:
        date_as_str = str(a_date)
        games = self.games_info[
            (self.games_info['gameDate'] <= date_as_str) &
            (self.games_info['opponentTeamAbbrev'] == team_abbrev)]
        games = games.head(n)
        r = games['gameId'].values.astype('str')
        return set(map(lambda f_v: int(f_v[5:]), r))

    def get_lines_for(self, base_date: datetime.date, how_many_games_back: int, team_abbrev: str) -> Optional[Formation]:
        """prediction of the lines that the 'away' team will use."""
        from ReinforcementLearning.NHL.playbyplay.game import Game

        assert (how_many_games_back >= 0)
        assert (team_abbrev in self.get_teams())
        ids = self.get_last_n_away_games_since(base_date, n=how_many_games_back, team_abbrev=team_abbrev)
        if len(ids) == 0:
            self.alogger.debug("I couldn't find %d games away for %s before %s" % (how_many_games_back, team_abbrev, base_date))
            return None
        lines_dict = {}
        for game_id in ids:
            entry_timestamp = datetime.datetime.now().timestamp()
            self.alogger.info("Processing game %d" % (game_id))
            if game_id in self.away_lines_per_game:
                self.alogger.debug(" => game cached")
                (g, result_as_list) = self.away_lines_per_game[game_id]
            else:
                g = Game(self, gameId=game_id)
                result_as_list = g.get_away_lines()
                self.away_lines_per_game[game_id] = (g, result_as_list)
            time_it_took = datetime.datetime.now().timestamp() - entry_timestamp
            self.alogger.debug("Fetching away lines took %.2f secs." % (time_it_took))
            for line_as_ids, line_as_types, secs_played in result_as_list:
                line_as_ids = tuple(map(g.player_id_to_name, line_as_ids))
                if line_as_ids in lines_dict:
                    # update number of seconds played
                    lines_dict[line_as_ids] = (line_as_types, lines_dict[line_as_ids][1] + secs_played)
                else:
                    # seed entry in dictionary
                    lines_dict[line_as_ids] = (line_as_types, secs_played)
        self.alogger.info("DONE")

        # for k, v in lines_dict.items():
        #     self.season.alogger.info(k, v)
        # ok, now sort by seconds played, keep top 4:
        flat_list = list(map(lambda x: (x[0], x[1][0], x[1][1]), lines_dict.items()))
        result_as_list = sorted(flat_list, key=lambda x: x[2], reverse=True)
        self.alogger.info("%d lines used consistently" % (len(result_as_list)))
        for a_line, a_cat, num_secs in result_as_list:
            self.alogger.info("%s played %.2f secs" % (a_line, num_secs))
        top_4_as_list = result_as_list[:4]
        self.alogger.info("Keeping top 4:")
        for a_line, a_cat, num_secs in top_4_as_list:
            self.alogger.info("%s played %.2f secs" % (a_line, num_secs))
        away_lines_names = list(map(lambda x: x[0], top_4_as_list))  # as names
        away_lines = list(map(lambda x: x[1], top_4_as_list))  # as categories
        return Formation(as_names=away_lines_names, as_categories=away_lines)

    def __str__(self):
        return "Season %d-%d" % (self.year_begin, self.year_end)
