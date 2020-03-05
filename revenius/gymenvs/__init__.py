"""Gym initarization."""
from gym.envs.registration import register


register(
    id='Reversi8x8-v0',
    entry_point='revenius.gymenvs.reversi:Reversi',
    kwargs={
        'plr_color': 'black',
        'opponent': 'random',
        'obs_type': 'numpy3c',
        'illegal_place_mode': 'lose',
        'board_size': 8,
    }
)
