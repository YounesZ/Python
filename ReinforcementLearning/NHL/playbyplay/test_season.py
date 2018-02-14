import unittest
import datetime
from random import randint
from os import path
from ReinforcementLearning.NHL.playbyplay.season import Season
from ReinforcementLearning.NHL.playbyplay.playbyplay_data import Game


class TestSeason(unittest.TestCase):
    """Testing definitions of Season's."""

    def setUp(self):
        """Initialization"""
        self.db_root = '/Users/luisd/dev/NHL_stats/data'
        self.repoCode = '/Users/luisd/dev/NHL_stats'
        self.repoModel = path.join(self.repoCode,
                              'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')
        self.season = Season(db_root=self.db_root, repo_model=self.repoModel, year_begin=2012)  # Season.from_year_begin(2012) # '20122013'

    def test_consistency(self):
        """Is there something weird with 'season'?"""
        home_team='MTL'
        for _ in range(10):
            the_month=randint(1,4)
            the_day=randint(1,25)
            base_date=datetime.date(year=2013, month=the_month, day=the_day)
            delta_in_days = randint(20, 40) # let's make this large enough for a game to always exist.
            print("[base: %s] Trying to fetch games for '%s', up until %d days before" % (base_date, home_team, delta_in_days))
            result = self.season.get_game_at_or_just_before(game_date=base_date, home_team_abbr=home_team, delta_in_days=delta_in_days)
            self.assertIsNotNone(result)
            game_id, d = result
            # ok. Now get game:
            the_game = Game(self.season, gameId=game_id)
            game_codes = the_game.df_wc['gcode'].unique()
            self.assertEqual(len(game_codes), 1) # must be unique
            game_code = game_codes[0]
            self.assertEqual(game_id, game_code) # must be the same one I asked initially
            # now, does it contain the team I asked for?
            self.assertEqual(len(the_game.df_wc["hometeam"].unique()), 1)
            self.assertEqual(the_game.df_wc["hometeam"].unique()[0], home_team,
                             "Expected: %s as home. But maybe it's away: %s?" % (home_team, the_game.df_wc["awayteam"].unique()[0]))



    def test_get_game_id(self):
        home_team='MTL'
        for _ in range(10):
            the_month=randint(1,5)
            the_day=randint(1,25)
            base_date=datetime.date(year=2013, month=the_month, day=the_day)
            # is there a game the SAME day I want?
            try:
                self.season.get_game_id(home_team_abbr='MTL', game_date=base_date)
                game_same_day = True
            except:
                game_same_day = False
            delta_in_days = randint(20, 40) # let's make this large enough for a game to always exist.
            result = self.season.get_game_at_or_just_before(game_date=base_date, home_team_abbr=home_team, delta_in_days=delta_in_days)
            self.assertIsNotNone(result)
            id, d = result
            # print("[base: %s] Fetched game %d, played on %s" % (base_date, id, d))
            delta_in_days = datetime.timedelta(days=delta_in_days)
            earliest_date = base_date - delta_in_days
            self.assertTrue(d >= earliest_date)
            if game_same_day:
                self.assertEqual(base_date, d)
            # now I get the Game, and check if it fetched the right team
            the_game = Game(self.season, gameId=id)
            self.assertEqual(the_game.home_team, home_team,
                             "[base: %s] Fetched game %d, played on %s. Away team: '%s'" % (base_date, id, d, the_game.away_team))
