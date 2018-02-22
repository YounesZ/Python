# -*- coding: utf-8 -*-
"""Test Games' Functionalities.

Attributes:

Todo:
    * Nothing for now.

"""
import unittest
import datetime
import random
from os import path
from ReinforcementLearning.NHL.playbyplay.game import Game
from ReinforcementLearning.NHL.playbyplay.season import Season


class TestGame(unittest.TestCase):
    """Testing definitions of Game's."""

    def setUp(self):
        """Initialization"""
        self.db_root = '/Users/luisd/dev/NHL_stats/data'
        self.repoCode = '/Users/luisd/dev/NHL_stats'

        self.repoModel = path.join(self.repoCode,
                              'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')

        # Now lets get game data
        self.season = Season(db_root=self.db_root, repo_model=self.repoModel, year_begin=2012)  # Season.from_year_begin(2012) # '20122013'
        # Montreal received Ottawa on march 13, 2013, let's convert game date to game code
        gameId = self.season.get_game_id(home_team_abbr='MTL', game_date=datetime.date(year=2013, month=3, day=13))
        self.a_game = Game(self.season, gameId=gameId)

    def test_shifts_differential(self):
        """Are the differential for lines properly calculated?"""

        days_before = 20
        for _ in range(10): # repeat this test 10 times
            random_date = datetime.date(year=2013, month=random.randint(1,4), day=random.randint(1,28))
            result = self.season.get_game_at_or_just_before(random_date, home_team_abbr='MTL', delta_in_days=days_before)
            if result is None:
                print("WARNING => No home game for MTL up to %d days before %s" % (days_before, random_date))
            else:
                random_game_id, game_date = result
                print("Examining game %d (from %s)" % (random_game_id, game_date))
                random_game = Game(self.season, gameId=random_game_id)
                df_differential = random_game.lineShifts.shifts['differential'].reset_index() # ?
                idxs_change_differential = df_differential.diff()[df_differential.diff().differential != 0].index.values
                for idx in idxs_change_differential[1:]: # skip fist one, because it's a NaN
                    self.assertNotEqual(random_game.lineShifts.shifts.iloc[idx]['GOAL'], 0,
                                        "Differential is %d but there was no goal scored" % (random_game.lineShifts.shifts.iloc[idx]['differential']))

    def test_shifts_goals_generate_differential(self):
        """Are the differential for lines properly calculated?"""

        days_before = 20
        for _ in range(10): # repeat this test 10 times
            random_date = datetime.date(year=2013, month=random.randint(1,4), day=random.randint(1,28))
            result = self.season.get_game_at_or_just_before(random_date, home_team_abbr='MTL', delta_in_days=days_before)
            if result is None:
                print("WARNING => No home game for MTL up to %d days before %s" % (days_before, random_date))
            else:
                random_game_id, game_date = result
                print("Examining game %d (from %s)" % (random_game_id, game_date))
                random_game = Game(self.season, gameId=random_game_id)
                df_goals = random_game.lineShifts.shifts['GOAL'].reset_index()
                idxs_goals = [a_val for a_val in df_goals[df_goals.GOAL != 0].index.values if a_val > 0]
                for idx in idxs_goals:
                    diff = random_game.lineShifts.shifts.iloc[idx]['differential']
                    diff_before = random_game.lineShifts.shifts.iloc[idx-1]['differential']
                    goals_now = random_game.lineShifts.shifts.iloc[idx]['GOAL']
                    goals_before = random_game.lineShifts.shifts.iloc[idx - 1]['GOAL']
                    expected_result = diff_before + goals_now
                    self.assertEqual(diff,expected_result,
                                        "\n%s\n [index: %d] Differential is %d but it should be => %d (== (diff before) %d + %d (goals now))" % (random_game.lineShifts.shifts.iloc[idx-2:idx+2][['GOAL', 'differential']], idx, diff, expected_result, diff_before, goals_now))
