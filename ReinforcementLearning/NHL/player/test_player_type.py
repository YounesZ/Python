import unittest
import random

from ReinforcementLearning.NHL.player.player_type import PlayerType

class TestPlayerType(unittest.TestCase):
    """Testing definitions of PlayerType's."""

    def setUp(self):
        """Initialization"""
        pass

    def test_values(self):
        """Specific values found in dataframes have to be respected, strictly."""

        # (1) are the values that I expect properly matched?
        dict_expected = { 0: PlayerType.DEFENSIVE, 1: PlayerType.OFFENSIVE, 2: PlayerType.NEUTRAL }
        for int_value, enum_value in dict_expected.items():
            self.assertEqual(PlayerType.from_int(value = int_value), enum_value,
                             "%d does not correspond with %s" % (int_value, enum_value))
        # and (2) probabilistically, are ANY other values rejected?
        some_ints_as_set = set(random.sample(range(1, 1000), 100)).difference({0,1,2})
        for a_value in some_ints_as_set:
            self.assertIsNone(PlayerType.from_int(a_value))


