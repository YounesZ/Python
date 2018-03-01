import datetime
import glob
import logging
import os
import pickle
from os import path

import numpy as np
from typing import Optional, List

from ReinforcementLearning.NHL.config import Config
from ReinforcementLearning.NHL.lines.category import CategoryFetcher
from ReinforcementLearning.NHL.lines.recommender import LineRecommender
from ReinforcementLearning.NHL.lines.valuation import QValuesFetcherFromGameData
from ReinforcementLearning.NHL.playbyplay.game import Game
from ReinforcementLearning.NHL.playbyplay.season import Season
from Utils.base import get_git_root


def get_teams_coach_performance(
        season: Season,
        team_abbr: str,
        maybe_a_starting_date: Optional[datetime.date],
        how_many_games: int,
        line_dict: dict,
        Qvalues) -> dict:
    """

    :param season: 
    :param team_abbr: 
    :param maybe_a_starting_date: 
    :param line_dict: 
    :param Qvalues: 
    :return: 
    """
    params = {
        "games_to_predict_away_lines": 7,
        "optimal_examine_num_first_lines": 5,  # None examines ALL
        "first_day_of_season": datetime.date(year=season.year_end, month=2, day=1),
        # "first_day_of_season": datetime.date(year=season.year_begin, month=12, day=1), # TODO: check why I don't find anything in season.year_begin?
    }

    assert (how_many_games > 0)
    last_game_date_accounted_for = None
    seconds = {  # result will be stored here.
        'does_not_match_optimal': 0,  # number of seconds the action of coach did not match the optimal
        'matches_optimal': 0,  # number of seconds the action of coach matched the optimal
        'unknown_adversary': 0  # number of seconds we can't determine the value of the action of coach
    }
    while how_many_games > 0:
        season.alogger.info("games left: %d; so far: %s" % (how_many_games, seconds))
        how_many_games -= 1
        if last_game_date_accounted_for is None:
            base_date = maybe_a_starting_date if maybe_a_starting_date is not None else params["first_day_of_season"]
            result = season.get_game_at_or_just_before(game_date=base_date, home_team_abbr=team_abbr, delta_in_days=20)
            if result is None:
                season.alogger.info("There is no game for '%s' just before %s" % (team_abbr, base_date))
                return seconds
            gameId, d = result
            base_date = base_date + datetime.timedelta(days=1)  # convenient for next cycle of computation
            last_game_date_accounted_for = base_date
        else:
            found = False
            while not found:
                result = season.get_game_at_or_just_before(game_date=base_date, home_team_abbr=team_abbr)
                # result should never be None, as at worst it will get the game it computed before.
                assert result is not None
                gameId, d = result
                base_date = base_date + datetime.timedelta(days=1)  # convenient for next cycle of computation
                found = (d != last_game_date_accounted_for)
            season.alogger.info("Fetched game %d, played on %s" % (gameId, d))
        data_for_a_game = Game(season, gameId)

        home_players = data_for_a_game.get_ids_of_home_players()
        if len(home_players) < 12:
            season.alogger.info(
                "Can't get enough info for home players (WEIRD!!). Ignoring game %d" % (data_for_a_game.gameId))
        else:
            # prediction of the lines that the 'away' team will use:
            formation_opt = season.get_lines_for(
                d - datetime.timedelta(days=1),
                how_many_games_back=params["games_to_predict_away_lines"],
                team_abbrev=data_for_a_game.away_team)
            if formation_opt is None:
                season.alogger.debug("Couldn't get a prediction of lines %s will use" % (data_for_a_game.away_team))
            else:
                formation = formation_opt
                away_lines_names = formation.as_names
                away_lines = formation.as_categories
                season.alogger.info(away_lines_names)
                season.alogger.info(away_lines)

                # === Now we get the indices in the Q-values tables corresponding to lines

                # Get lines and translate them
                playersCode = data_for_a_game.encode_line_players()
                linesCode = np.array([[data_for_a_game.recode_line(line_dict, a) for a in b] for b in playersCode])

                # Get the Q-value for that specific line
                iShift = 0  # First shift
                lineShifts = data_for_a_game.lineShifts.as_df(team='both',
                                                              equal_strength=data_for_a_game.shifts_equal_strength,
                                                              regular_time=data_for_a_game.shifts_regular_time,
                                                              min_duration=20)

                player_classes = data_for_a_game.players_classes_mgr.get(equal_strength=True, regular_time=True,
                                                                         min_duration=20,
                                                                         nGames=30)  # TODO: why these parameters?
                plList = list(player_classes.loc[lineShifts['playersID'].iloc[iShift][0]]['firstlast'].values) + \
                         list(player_classes.loc[lineShifts['playersID'].iloc[iShift][1]]['firstlast'].values)
                diff = data_for_a_game.recode_differential(lineShifts.iloc[iShift].differential)
                period = data_for_a_game.recode_period(lineShifts.iloc[iShift].period)
                q_values = Qvalues[period, diff, linesCode[iShift, 0], linesCode[iShift, 1]]
                season.alogger.info(
                    '[diff = %d, period = %d] First shift: \n\thome team: %s, %s, %s \n\taway team: %s, %s, %s \n\tQvalue: %.2f' % (
                        diff, period, plList[0], plList[1], plList[2], plList[3], plList[4], plList[5], q_values))

                q_values_fetcher_from_game_data = QValuesFetcherFromGameData(game_data=data_for_a_game,
                                                                             lines_dict=line_dict, q_values=Qvalues)

                line_rec = LineRecommender(
                    game=data_for_a_game,
                    player_category_fetcher=CategoryFetcher(data_for_game=data_for_a_game),
                    q_values_fetcher=q_values_fetcher_from_game_data)

                home_lines_rec = line_rec.recommend_lines_maximize_average(
                    home_team_players_ids=data_for_a_game.get_ids_of_home_players(),
                    away_team_lines=away_lines,
                    examine_max_first_lines=params["optimal_examine_num_first_lines"])
                season.alogger.info(home_lines_rec)

                season.alogger.info(data_for_a_game.formation_ids_to_str(home_lines_rec))

                # let's examine actual decisions and how it compares with optimal:
                for data in lineShifts[['home_line', 'away_line', 'iceduration']].itertuples():
                    home_line = data.home_line
                    away_line = data.away_line
                    num_seconds = data.iceduration
                    away_line_cats = data_for_a_game.classes_of_line(away_line)
                    if None in away_line_cats:
                        season.alogger.info("Can't get category of one of away players")
                    else:
                        away_line_cats = tuple(np.sort(away_line_cats))
                        if away_line_cats not in away_lines:
                            # season.alogger.debug("%s (categories %s): no optimal calculated" % (away_line, away_line_cats))
                            seconds['unknown_adversary'] += num_seconds
                        else:
                            idx_of_away = away_lines.index(away_line_cats)
                            cats_of_optimal = data_for_a_game.classes_of_line(home_lines_rec[idx_of_away])
                            home_line_cats = data_for_a_game.classes_of_line(home_line)
                            if set(cats_of_optimal) == set(home_line_cats):
                                seconds['matches_optimal'] += num_seconds
                            else:
                                seconds['does_not_match_optimal'] += num_seconds
    return seconds

class Coach_Evaluator(object):

    def __init__(self, alogger: logging.Logger):
        self.alogger = alogger
        self.base_dir = os.path.join(Config().data_dir, "coaches_perf")
        self.alogger.debug("Read info from '%s'" % (self.base_dir))

    def evaluate_mtl_coach(self, season_year_begin: int, n_games: int):
        self.evaluate_all_coaches(season_year_begin, teams_opt=['MTL'], n_games=n_games)

    def evaluate_all_coaches(self, season_year_begin: int, teams_opt: Optional[List[str]], n_games: int):
        from ReinforcementLearning.NHL.playbyplay.state_space_data import HockeySS

        """Initialization"""
        os.makedirs(self.base_dir, exist_ok=True)
        my_config=Config()
        self.alogger.debug("Data configured to be in '%s'" % (my_config.data_dir))

        db_root = my_config.data_dir
        repoCode = get_git_root()

        repoModel = path.join(repoCode,
                                   'ReinforcementLearning/NHL/playerstats/offVSdef/Automatic_classification/MODEL_perceptron_1layer_10units_relu')

        season = Season(self.alogger, db_root=db_root, year_begin=season_year_begin, repo_model=repoModel)

        # Line translation table
        linedict  = HockeySS(db_root)
        linedict.make_line_dictionary()
        linedict  = linedict.line_dictionary

        # Load the Qvalues table
        Qvalues = \
        pickle.load(open(path.join(repoCode, 'ReinforcementLearning/NHL/playbyplay/data/stable/RL_action_values.p'), 'rb'))[
            'action_values']

        # Visualize it dimensions (period x differential x away line's code x home line's code)
        print('Q-table dimensions: ', Qvalues.shape)

        # for what teams will we run this calculation?
        calc_teams = season.get_teams() if teams_opt is None else teams_opt
        for a_team in calc_teams:
            season.alogger.debug("=============> calculating %s" % (a_team))
            seconds = get_teams_coach_performance(
                season,
                team_abbr = a_team,
                maybe_a_starting_date=None,
                line_dict=linedict,
                Qvalues=Qvalues,
                how_many_games=n_games)
            season.alogger.debug(seconds)

            if seconds["does_not_match_optimal"] == seconds["matches_optimal"] == 0:
                season.alogger.info("[team: '%s'] No evidence for coach to be evaluated on." % (a_team))
            else:
                total_secs = seconds['matches_optimal'] + seconds['does_not_match_optimal']
                season.alogger.info("['%s'] Home coach's score is %d (secs. optimal) / %d (secs. total) = %.2f (in [0,1])" %
                      (a_team, seconds['matches_optimal'], total_secs, seconds['matches_optimal'] / total_secs))
            file_to_save = os.path.join(self.base_dir, a_team + ".pkl")
            self.alogger.debug("Saving data for '%s' in file %s" % (a_team, file_to_save))
            with open(file_to_save, 'wb') as dict_file:
                pickle.dump(seconds, dict_file)
            self.alogger.debug("DONE")

    def read_and_display_coaches_evals(self) -> dict:
        os.chdir(self.base_dir)
        result = {}
        for filename in glob.glob("*.pkl"):
            # print(filename)
            the_split = os.path.splitext(filename)
            name = the_split[0]
            extension = the_split[1]
            with open(filename, 'rb') as pkl_file:
                a_perf = pickle.load(pkl_file)
                if a_perf['matches_optimal'] == a_perf['does_not_match_optimal'] == 0:
                    grade = 0.0
                    print("%s: %s" % (name, a_perf))
                else:
                    grade = a_perf['matches_optimal'] / (a_perf['matches_optimal'] + a_perf['does_not_match_optimal'])
                    print("%s: %s; performance is %.2f" % (name,a_perf, grade))
                result[name] = grade
        return result

if __name__ == '__main__':
    from Utils.base import get_logger
    common_logger = get_logger(name="common_logger", debug_log_file_name="common_logger.log")
    print("Debug will be written in {}".format(common_logger.handlers[1].baseFilename))

    coaches_evaluator = Coach_Evaluator(common_logger)
    coaches_evaluator.evaluate_all_coaches(season_year_begin = 2012, teams_opt=None, n_games=2) # 10) # ['PHX'])
