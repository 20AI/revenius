"""Opponent module."""
import numpy as np

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


def make_error_policy(opp):
    """Make error policy."""
    if opp == 'dqn':
        def error_policy(state, plr_color):
            raise('Please do set_model first.')
    return error_policy


def make_dqn_policy(model, board_size):
    """Make dqn policy."""
    def dqn_policy(state, plr_color):
        possible_places = get_possible_acts(state, plr_color)
        q_values = model.compute_q_values(
            state.reshape((1,)+state.shape))
        # action = model.policy.select_action(q_values)
        if len(possible_places) == 0:
            return board_size**2 + 1
        possible_q = np.zeros(len(q_values))
        for _i, _q in enumerate(q_values):
            if _i not in possible_places:
                _q = np.min(q_values)
            possible_q[_i] = _q
        return np.argmax(possible_q)
    return dqn_policy

#
# class DQNOpponent():
#     """Opponent class."""
#
#     def __init__(self, model, board_size):
#         """initarize."""
#         self.model = model
#         self.board_size = board_size
#
#     def __call__(self, state, plr_color):
#         """Call."""
#         possible_places = get_possible_acts(state, plr_color)
#         q_values = self.model.compute_q_values(
#             state.reshape((1,)+state.shape))
#         # action = model.policy.select_action(q_values)
#         if len(possible_places) == 0:
#             return self.board_size**2 + 1
#         possible_q = np.zeros(len(q_values))
#         for _i, _q in enumerate(q_values):
#             if _i not in possible_places:
#                 _q = np.min(q_values)
#             possible_q[_i] = _q
#
#         return np.argmax(possible_q)
