import itertools
from abc import abstractmethod
from typing import List, Dict, Tuple, Set

from ReinforcementLearning.NHL.player.player_type import PlayerType

class QValuesFetcher(object):
    def lines_to_index(home_line: List[PlayerType], away_line: List[PlayerType], period: int, diff: int) -> str:
        return "period%d_diff%d_%s" % (period, diff, ''.join(list(map(str, home_line + away_line))))

    def __init__(self):
        pass

    @abstractmethod
    def get(self, period, diff, home_line: List[PlayerType], away_line: List[PlayerType]) -> float:
        raise NotImplementedError


class QValuesFetcherFromDict(QValuesFetcher):
    def __init__(self, a_dict: Dict):
        QValuesFetcher.__init__(self)
        self.dict = a_dict

    @classmethod
    def from_tuples(cls, q_value_tuples: List[Tuple[List[PlayerType], List[PlayerType], float]]):
        q_value_dict = {}
        for v in q_value_tuples:
            home_line, away_line, q_value = v
            for home_line_comb in list(set(itertools.permutations(home_line))):
                for away_line_comb in list(set(itertools.permutations(away_line))):
                    q_value_dict[QValuesFetcher.lines_to_index(home_line_comb, away_line_comb, period=1, diff=0)] = q_value # fix period and diff. All good, this is just for testing purposes.
        print("Q value dictionary has %d entries" % (len(q_value_dict)))
        return cls(a_dict=q_value_dict)

    def get(self, period, diff, home_line: Set[PlayerType], away_line: Set[PlayerType]) -> float:
        # TODO: dafuq!!!!!!!!!
        from random import uniform

        return uniform(1, 30)
        # return self.dict[QValuesFetcher.lines_to_index(list(home_line), list(away_line))]


class QValuesFetcherFromGameData(QValuesFetcher):

    def __init__(self, game_data, lines_dict, q_values):
        QValuesFetcher.__init__(self)
        self.game_data  = game_data
        self.lines_dict = lines_dict
        self.q_values = q_values

    def get(self, period, diff, home_line: List[PlayerType], away_line: List[PlayerType]) -> float:
        return self.q_values[
                period,
                diff,
                self.game_data.recode_line(self.lines_dict, home_line),
                self.game_data.recode_line(self.lines_dict, away_line)]


# ###########################################################################################
# ###########################################################################################
# ###########################################################################################

