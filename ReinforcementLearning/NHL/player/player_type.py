from enum import IntEnum

class PlayerType(IntEnum):
    """"Type of a player."""

    # TODO: find a way to tie this with the dataframes containing 'int's for player classes.
    DEFENSIVE = 0
    OFFENSIVE = 1
    NEUTRAL = 2

    @classmethod
    def from_int(cls, value: int):
        """ Returns the player type whose value is this int, if found. Otherwise returns None."""
        for name, member in cls.__members__.items():
            if member == value: # this comparison OK because this is an IntEnum.
                return member
        return None
