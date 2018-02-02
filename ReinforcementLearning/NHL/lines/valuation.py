import itertools
from abc import abstractmethod
from typing import List, Dict, Tuple, Set

from ReinforcementLearning.NHL.player.player_type import PlayerType

class QValuesFetcher(object):
    def lines_to_index(home_line: List[PlayerType], away_line: List[PlayerType]) -> str:
        return ''.join(list(map(str, home_line + away_line)))

    def __init__(self):
        pass

    @abstractmethod
    def get(self, home_line: List[PlayerType], away_line: List[PlayerType]) -> float:
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
                    q_value_dict[QValuesFetcher.lines_to_index(home_line_comb, away_line_comb)] = q_value
        print("Q value dictionary has %d entries" % (len(q_value_dict)))
        return cls(a_dict=q_value_dict)

    def get(self, home_line: Set[PlayerType], away_line: Set[PlayerType]) -> float:
        # TODO: dafuq!!!!!!!!!
        from random import uniform

        return uniform(1, 30)
        # return self.dict[QValuesFetcher.lines_to_index(list(home_line), list(away_line))]

# ###########################################################################################
# ###########################################################################################
# ###########################################################################################

