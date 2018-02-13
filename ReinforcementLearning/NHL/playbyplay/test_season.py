import unittest
import datetime
from random import randint
from os import path
from ReinforcementLearning.NHL.playbyplay.season import Season


class TestSeason(unittest.TestCase):
    """Testing definitions of Season's."""

    def setUp(self):
        """Initialization"""
        self.db_root = '/Users/luisd/dev/NHL_stats/data'
        self.repoCode = '/Users/luisd/dev/NHL_stats'
        self.repoModel = path.join(self.repoCode,
                              'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')
        self.season = Season(db_root=self.db_root, repo_model=self.repoModel, year_begin=2012)  # Season.from_year_begin(2012) # '20122013'

    def test_get_game_id(self):
        # self.season.get_game_id(home_team_abbr='MTL', game_date=datetime.date(year=2013, month=3, day=13))
        base_date=datetime.date(year=2013, month=3, day=13)
        # is there a game the SAME day I want?
        try:
            self.season.get_game_id(home_team_abbr='MTL', game_date=base_date)
            game_same_day = True
        except:
            game_same_day = False
        for _ in range(10):
            delta_in_days = randint(1, 20)
            result = self.season.get_game_at_or_just_before(game_date=base_date, home_team_abbr='MTL', delta_in_days=delta_in_days)
            self.assertIsNotNone(result)
            id, d = result
            delta_in_days = datetime.timedelta(days=delta_in_days)
            earliest_date = base_date - delta_in_days
            self.assertTrue(d >= earliest_date)
            if game_same_day:
                self.assertEqual(base_date, d)
