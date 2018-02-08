from ReinforcementLearning.NHL.player.player_type import PlayerType


def get_class_of_player_by_id(data_for_game, player_id: int) -> PlayerType:
    try:
        return PlayerType.from_int(int(data_for_game.player_classes[data_for_game.player_classes.index == player_id]["class"]))
    except:
        raise IndexError("Player %d did not play on this game" % (player_id))

class CategoryFetcher(object):

    def __init__(self, data_for_game, cache_ids: bool = True):
        self.data_for_game = data_for_game
        self.cache_ids = cache_ids
        self.cache_player_ids = {} # cache for ids

    def category_of_player(self, player_id: int) -> PlayerType:
        if self.cache_ids:
            if player_id not in self.cache_player_ids:
                self.cache_player_ids[player_id] = get_class_of_player_by_id(self.data_for_game, player_id)
            return self.cache_player_ids[player_id]
        else:
            return get_class_of_player_by_id(self.data_for_game, player_id)
