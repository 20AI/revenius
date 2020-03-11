"""Reversi environment."""
from six import StringIO
import sys
import time

import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding
from IPython.display import clear_output

from .util import make_random_policy
from .util import get_possible_acts
from .util import make_place
from .util import is_pass_act
from .util import is_resign_act
from .util import is_valid
from .util import is_finished


class Reversi(gym.Env):
    """Reversi environment."""

    BLACK = 0
    WHITE = 1
    metadata = {'render.modes': ['ansi', 'human']}

    def __init__(
            self, plr_color, opponent, obs_type,
            illegal_place_mode, board_size):
        """Initarixe."""
        assert isinstance(board_size, int) and board_size >= 1,\
            'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        colormap = {
            'black': self.BLACK,
            'white': self.WHITE,
        }
        try:
            self.plr_color = colormap[plr_color]
        except KeyError:
            raise error.Error(
                "plr_color must be 'black' or 'white', not {}".format(
                    plr_color))

        self.opp = opponent

        assert obs_type in ['numpy3c']
        self.obs_type = obs_type

        assert illegal_place_mode in ['lose', 'raise']
        self.illegal_place_mode = illegal_place_mode

        if self.obs_type != 'numpy3c':
            raise error.Error(
                'Unsupported observation type: {}'.format(self.obs_type))

        self.action_space = spaces.Discrete(self.board_size ** 2 + 2)
        obs = self.reset()
        self.obs_space = spaces.Box(
            np.zeros(obs.shape), np.ones(obs.shape))

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        if isinstance(self.opp, str):
            if self.opp == 'random':
                self.opp_policy = make_random_policy(
                    self.np_random, self.board_size)
            else:
                raise error.Error(
                    'Unrecognized opponent policy {}'.format(self.opp))
        else:
            self.opp_policy = self.opp

        return [seed]

    def reset(self):
        """Define reset."""
        self.state = np.zeros((3, self.board_size, self.board_size))
        center_l = int(self.board_size/2-1)
        center_r = int(self.board_size/2)
        self.state[2, :, :] = 1.0
        self.state[2, (center_l):(center_r+1), (center_l):(center_r+1)] = 0
        self.state[0, center_r, center_l] = 1
        self.state[0, center_l, center_r] = 1
        self.state[1, center_l, center_l] = 1
        self.state[1, center_r, center_r] = 1
        self.to_play = self.BLACK
        self.possible_actions = get_possible_acts(
            self.state, self.to_play)
        self.done = False

        if self.plr_color != self.to_play:
            a = self.opp_policy(self.state)
            make_place(self.state, a, self.BLACK)
            self.to_play = self.WHITE
        return self.state

    def _step_return(self, reward, done):
        return self.state, reward, done, {}

    def step(self, action):
        """Define step."""
        assert self.to_play == self.plr_color
        if self.done:
            return self._step_return(0., True)
        if is_pass_act(self.board_size, action):
            pass
        elif is_resign_act(self.board_size, action):
            return self._step_return(-1., True)
        elif not is_valid(self.state, action, self.plr_color):
            if self.illegal_place_mode == 'raise':
                raise
            elif self.illegal_place_mode == 'lose':
                self.done = True
                return self._step_return(-1., True)
            else:
                raise error.Error(
                    'Unsupported illegal place action: {}'.format(
                        self.illegal_place_mode))
        else:
            make_place(self.state, action, self.plr_color)

        _a = self.opp_policy(self.state, 1 - self.plr_color)

        if _a is not None:
            if is_pass_act(self.board_size, _a):
                pass
            elif is_resign_act(self.board_size, _a):
                return self._step_return(1., True)
            elif not is_valid(self.state, _a, 1 - self.plr_color):
                if self.illegal_place_mode == 'raise':
                    raise
                elif self.illegal_place_mode == 'lose':
                    self.done = True
                    return self._step_return(1., True)
                else:
                    raise error.Error(
                        'Unsupported illegal place action: {}'.format(
                            self.illegal_place_mode))
            else:
                make_place(self.state, _a, 1 - self.plr_color)

        self.possible_actions = get_possible_acts(
            self.state, self.plr_color)
        reward = is_finished(self.state)
        if self.plr_color == self.WHITE:
            reward = - reward
        self.done = reward != 0

        return self._step_return(reward, self.done)

    def render(self, mode='human', close=False):
        """Define render."""
        if not close:
            board = self.state
            _, _h, _w = board.shape
            outfile = StringIO() if mode == 'ansi' else sys.stdout

            outfile.write(' ' * 7)
            outfile.write(' '.join([str(j+1)+'  | ' for j in range(_w)]))
            outfile.write('\n')
            outfile.write(' ' * 5 + '-' * (_w * 6 - 1) + '\n')
            for i in range(_h):
                outfile.write(' ' + str(i + 1) + '  |')
                for j in range(_w):
                    if board[0, i, j] == 1:
                        outfile.write('  B  ')
                    elif board[1, i, j] == 1:
                        outfile.write('  W  ')
                    else:
                        outfile.write('  .  ')
                    outfile.write('|')
                outfile.write('\n')
                outfile.write(' ' + '-' * (_w * 7 - 1) + '\n')

            if mode != 'human':
                if mode == 'ipython':
                    time.sleep(0.1)
                    clear_output(wait=True)
                return outfile
