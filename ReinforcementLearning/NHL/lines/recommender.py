import itertools
import datetime

from typing import List, Callable, Set, Optional

from ReinforcementLearning.NHL.player.player_type import PlayerType
from ReinforcementLearning.NHL.lines.category import CategoryFetcher
from ReinforcementLearning.NHL.lines.valuation import QValuesFetcher
from ReinforcementLearning.NHL.playbyplay.game import Game

class LineRecommender(object):

    def __init__(self, game: Game, player_category_fetcher: CategoryFetcher, q_values_fetcher: QValuesFetcher):
        self.player_category_fetcher=player_category_fetcher
        self.q_values_fetcher=q_values_fetcher
        self.game = game

    def recommend_lines(
            self,
            home_team_players_ids: Set[int],
            away_team_lines: List[List[PlayerType]],
            comb_q_values: Callable[[List[float]], float],
            examine_max_first_lines: Optional[int] = None) -> List[List[int]]:
        """
        Builds optimal lines for home team.
        Args:
            home_team_players_ids: a list of id's for home team.
            away_team_lines: 4 lines of away team. Each line is composed of 3 players, represented by their class (0 = defensive, 2 = offensive, 1 = neither)
            q_values: a function that given 2 lines (represented by CLASSES of players) returns the q-value of the home team line.
            comb_q_values: how to combine the q-values in order to obtain a "fitness" for a formation (larger is better).
        Returns:

        """
        def cats_from_ids(home_line_1_ids: List[int]) -> List[PlayerType]:
            return list(map(self.player_category_fetcher.category_of_player, home_line_1_ids))

        def formation_used_more_often(a_formation: List[List[int]], another_formation: Optional[List[List[int]]]):
            "Is one formtion used more often than another?"
            if another_formation is None:
                return True
            else:
                # TODO: get 'temps de glace' for each one of the lines, sum thenm up.
                return False # True

        def get_q_value(home_line, away_line) -> float:
            "let's see how these lines fare against each other in different circumstances"
            all_period_diffs = itertools.product(range(3), range(-2, 3))
            return sum(
                [self.q_values_fetcher.get(period_diff[0], period_diff[1], home_line, away_line)
                 for period_diff in all_period_diffs])

        assert len(home_team_players_ids) == len(set(home_team_players_ids)), "There are repeated ids in the home team"
        assert len(home_team_players_ids) >= 12, "Only %d players available for home team" % (len(home_team_players_ids))
        assert len(away_team_lines) == 4, "I need a formation (ie, 4 lines) for away team"

        away_line_1, away_line_2, away_line_3, away_line_4 = away_team_lines
        best_fitness = None
        best_formation = None
        how_many_first_lines_tried = 0
        #
        entry_timestamp = datetime.datetime.now().timestamp()
        for home_line_1_ids in itertools.combinations(home_team_players_ids, 3):
            best_formation_found = False
            home_line_1 = cats_from_ids(home_line_1_ids)
            for home_line_2_ids in itertools.combinations(home_team_players_ids -  set(home_line_1_ids), 3):
                home_line_2 = cats_from_ids(home_line_2_ids)
                for home_line_3_ids in itertools.combinations(
                                        home_team_players_ids - set(home_line_1_ids) - set(home_line_2_ids), 3):
                    home_line_3 = cats_from_ids(home_line_3_ids)
                    for home_line_4_ids in itertools.combinations(
                                        home_team_players_ids - set(home_line_1_ids) - set(home_line_2_ids) - set(home_line_3_ids), 3):
                        home_formation = [home_line_1_ids, home_line_2_ids, home_line_3_ids, home_line_4_ids]

                        # print(home_formation)
                        # get lines with CATEGORY of players
                        home_line_4 = cats_from_ids(home_line_4_ids)
                        # ok, then; let's see how these lines fare against each other in different circumstances:

                        qs = [get_q_value(home_line_1, away_line_1),
                              get_q_value(home_line_2, away_line_2),
                              get_q_value(home_line_3, away_line_3),
                              get_q_value(home_line_4, away_line_4)]
                        fitness = comb_q_values(qs)
                        # print(qs)
                        if (best_fitness is None) or \
                                (fitness > best_fitness) or \
                                ((fitness == best_fitness) and formation_used_more_often(home_formation, best_formation)):
                            best_fitness = fitness
                            best_formation = home_formation
                            best_formation_found = True
                            print("Best fitness: %.2f by formation \n%s" % (best_fitness, self.game.formation_ids_to_str(best_formation)))

            how_many_first_lines_tried += 1
            all_done = (examine_max_first_lines is not None and (how_many_first_lines_tried >= examine_max_first_lines))
            if (best_formation_found or all_done):
                time_it_took = datetime.datetime.now().timestamp() - entry_timestamp
                time_per_cycle = time_it_took / how_many_first_lines_tried
                print("=======> Took %.2f secs. to look at %d first-lines; I think we have around %.2f secs. to go" % (
                time_it_took, how_many_first_lines_tried, (220 - how_many_first_lines_tried) * time_per_cycle))
            if all_done:
                break

        print("ALL DONE!!!!!!")
        print("================================")
        print("Best fitness: %.2f by formation \n%s, to play against \n%s" % (best_fitness, best_formation, away_team_lines))
        print("================================")
        return best_formation

    def recommend_lines_maximize_average(
            self,
            home_team_players_ids: Set[int],
            away_team_lines: List[List[PlayerType]],
            examine_max_first_lines: Optional[int] = None) -> List[List[int]]:
        return self.recommend_lines(home_team_players_ids, away_team_lines, comb_q_values=(lambda a_list: sum(a_list) / len(a_list)), examine_max_first_lines=examine_max_first_lines)

    def recommend_lines_maximize_max(
            self,
            home_team_players_ids: Set[int],
            away_team_lines: List[List[PlayerType]],
            examine_max_first_lines: Optional[int] = None) -> List[List[int]]:
        return self.recommend_lines(home_team_players_ids, away_team_lines, comb_q_values=(lambda a_list: max(a_list)), examine_max_first_lines=examine_max_first_lines)
# ###########################################################################################
# ###########################################################################################
# ###########################################################################################

