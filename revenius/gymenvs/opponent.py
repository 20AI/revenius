"""Opponent module."""

from .util import get_possible_acts


def make_random_policy(np_random, board_size):
    """Make random policy."""
    def random_policy(state, plr_color):
        possible_places = get_possible_acts(state, plr_color)
        if len(possible_places) == 0:
            return board_size**2 + 1
        a = np_random.randint(len(possible_places))
        return possible_places[a]
    return random_policy
