import pickle
from os import path

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