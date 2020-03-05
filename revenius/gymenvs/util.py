"""Util module."""
import numpy as np


def is_resign_act(board_size, action):
    """Check if the action is resign."""
    return action == board_size ** 2


def is_pass_act(board_size, action):
    """Check if the action is pass."""
    return action == board_size ** 2 + 1


def get_possible_acts(board, plr_color):
    """Get possible actions."""
    actions = []
    _d = board.shape[-1]
    opp_color = 1 - plr_color
    for pos_x in range(_d):
        for pos_y in range(_d):
            if (board[2, pos_x, pos_y] == 0):
                continue
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if(dx == 0 and dy == 0):
                        continue
                    nx = pos_x + dx
                    ny = pos_y + dy
                    n = 0
                    if (nx not in range(_d) or ny not in range(_d)):
                        continue
                    while(board[opp_color, nx, ny] == 1):
                        tmp_nx = nx + dx
                        tmp_ny = ny + dy
                        if (tmp_nx not in range(_d)
                                or tmp_ny not in range(_d)):
                            break
                        n += 1
                        nx += dx
                        ny += dy
                    if(n > 0 and board[plr_color, nx, ny] == 1):
                        actions.append(pos_x * _d + pos_y)
    if len(actions) == 0:
        actions = [_d**2 + 1]
    return actions


def valid_reverse_opp(board, coords, plr_color):
    """Check if there is a reversible location."""
    _d = board.shape[-1]
    opp_color = 1 - plr_color
    pos_x, pos_y = coords[:2]
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if(dx == 0 and dy == 0):
                continue
            nx = pos_x + dx
            ny = pos_y + dy
            _n = 0
            if (nx not in range(_d) or ny not in range(_d)):
                continue
            while(board[opp_color, nx, ny] == 1):
                tmp_nx = nx + dx
                tmp_ny = ny + dy
                if (tmp_nx not in range(_d) or tmp_ny not in range(_d)):
                    break
                _n += 1
                nx += dx
                ny += dy
            if(_n > 0 and board[plr_color, nx, ny] == 1):
                return True
    return False


def is_valid(board, action, plr_color):
    """Check valid place."""
    coords = act2coord(board, action)
    if board[2, coords[0], coords[1]] == 1:
        if valid_reverse_opp(board, coords, plr_color):
            return True
        else:
            return False
    else:
        return False


def make_place(board, action, plr_color):
    """Make palce."""
    coords = act2coord(board, action)

    _d = board.shape[-1]
    opp_color = 1 - plr_color
    pos_x, pos_y = coords[:2]

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if(dx == 0 and dy == 0):
                continue
            nx = pos_x + dx
            ny = pos_y + dy
            _n = 0
            if (nx not in range(_d) or ny not in range(_d)):
                continue
            while(board[opp_color, nx, ny] == 1):
                tmp_nx = nx + dx
                tmp_ny = ny + dy
                if (tmp_nx not in range(_d) or tmp_ny not in range(_d)):
                    break
                _n += 1
                nx += dx
                ny += dy
            if(_n > 0 and board[plr_color, nx, ny] == 1):
                nx = pos_x + dx
                ny = pos_y + dy
                while(board[opp_color, nx, ny] == 1):
                    board[2, nx, ny] = 0
                    board[plr_color, nx, ny] = 1
                    board[opp_color, nx, ny] = 0
                    nx += dx
                    ny += dy
                board[2, pos_x, pos_y] = 0
                board[plr_color, pos_x, pos_y] = 1
                board[opp_color, pos_x, pos_y] = 0
    return board


def coord2act(board, coords):
    """Convert coordinate to action."""
    return coords[0] * board.shape[-1] + coords[1]


def act2coord(board, action):
    """Convert action to coordinate."""
    return action // board.shape[-1], action % board.shape[-1]


def is_finished(board):
    """Chack game is finished."""
    _d = board.shape[-1]

    plr_score_x, plr_score_y = np.where(board[0, :, :] == 1)
    plr_score = len(plr_score_x)
    opp_score_x, opp_score_y = np.where(board[1, :, :] == 1)
    opp_score = len(opp_score_x)
    if plr_score == 0:
        return -1
    elif opp_score == 0:
        return 1
    else:
        free_x, free_y = np.where(board[2, :, :] == 1)
        if free_x.size == 0:
            if plr_score > (_d**2)/2:
                return 1
            elif plr_score == (_d**2)/2:
                return 1
            else:
                return -1
        else:
            return 0
    return 0


def make_random_policy(np_random, board_size):
    """Make random policy."""
    def random_policy(state, plr_color):
        possible_places = get_possible_acts(state, plr_color)
        if len(possible_places) == 0:
            return board_size**2 + 1
        a = np_random.randint(len(possible_places))
        return possible_places[a]
    return random_policy
