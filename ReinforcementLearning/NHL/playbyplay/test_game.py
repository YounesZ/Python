import unittest
from os import path
from ReinforcementLearning.NHL.playbyplay.playbyplay_data import Game
from ReinforcementLearning.NHL.playbyplay.season import Season


class TestGame(unittest.TestCase):
    """Testing definitions of Game's."""

    def setUp(self):
        """Initialization"""
        self.db_root = '/Users/luisd/dev/NHL_stats/data'
        self.repoCode = '/Users/luisd/dev/NHL_stats'
        # Montreal received Ottawa on march 13, 2013, let's convert game date to game code
        gameId = Season.get_game_id(db_root=self.db_root, home_team_abbr='MTL', date_as_str='2013-03-13')

        self.repoModel = path.join(self.repoCode,
                              'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')

        # Now lets get game data
        self.season = Season(year_begin=2012)  # Season.from_year_begin(2012) # '20122013'
        self.mtlott = Game(self.season, gameId=gameId)

    def test1(self):
        # Visualize all player's classes: 0=def, 1=off, 2=neutral
        players_classes = self.mtlott.pull_players_classes_from_repo_address(self.repoModel, number_of_games=30)
        self.assertTrue(players_classes.equals(self.mtlott.player_classes))

