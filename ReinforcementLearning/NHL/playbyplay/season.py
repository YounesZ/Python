import pickle
import datetime

from os import path
from typing import Tuple, Optional

class Season:
    """Encapsulates all elements for a season."""

    def __init__(self, db_root: str, repo_model: str, year_begin: int):
        self.db_root    =   db_root
        self.repo_model = repo_model
        self.year_begin =   year_begin
        self.year_end   =   self.year_begin + 1

        # List games and load season data
        self.repoPbP    =   path.join(self.db_root, 'PlayByPlay')
        self.repoPSt    =   path.join(self.db_root, "PlayerStats", "player")
        # Get data - long
        dataPath        =   path.join(self.repoPbP, 'Season_%d%d' % (self.year_begin, self.year_end),'converted_data.p')
        self.dataFrames =   pickle.load(open(dataPath, 'rb'))
        # Get game IDs
        self.games_id   =   self.dataFrames['playbyplay'].drop_duplicates(subset=['season', 'gcode'], keep='first')[['season', 'gcode', 'refdate', 'hometeam', 'awayteam']]
        #
        self.games_info = pickle.load(open(path.join(self.db_root, 'processed', 'gamesInfo.p'), 'rb'))
        self.games_info = self.games_info[
            (self.games_info['gameDate'] >= ('%d-09-01' % (self.year_begin))) & (self.games_info['gameDate'] <= ('%d-07-01' % (self.year_end)))] # take games only for this season
        self.games_info = self.games_info.sort_values(by=['gameDate'], ascending=False)


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

    def __str__(self):
        return "Season %d-%d" % (self.year_begin, self.year_end)
