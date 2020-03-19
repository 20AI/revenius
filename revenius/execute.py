"""Execute module."""
import random

import numpy as np
import gym

import revenius.gymenvs as gymenvs
from revenius.gymenvs.util import get_possible_acts


def random_walk(env, n_episode=5):
    """Execute random walk."""
    env.reset()
    for i_episode in range(n_episode):
        observation = env.reset()
        for t in range(100):
            enables = env.possible_actions
            if len(enables) == 0:
                action = env.board_size**2 + 1
            else:
                action = random.choice(enables)
            observation, reward, done, info = env.step(action)
            env.render()
            print('reward :', reward)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                black_score = len(np.where(
                    env.state[0, :, :] == 1)[0])
                print(black_score)
                break
    return env


def _get_dqn_action(dqn, env, observation):
    q_values = dqn.compute_q_values(
        observation.reshape((1,)+env.obs_space.shape))
    action = dqn.policy.select_action(q_values)
    return action


def _show_possible_acts(observation, possible_acts):
    from revenius.gymenvs.util import act2coord
    _coord = [act2coord(observation, _p) for _p in possible_acts]
    print("Possible acts :")
    print([(_c[0]+1, _c[1]+1) for _c in _coord])


def human_vs_dqn(dqn, plr_color=0):
    """Execute human vs dqn."""
    assert gymenvs
    from revenius.gymenvs.util import coord2act

    env = gym.make('Reversi8x8_dqn-v0')
    env.set_model(dqn)

    observation = env.reset()
    done = False
    while not done:
        env.render()
        possible_acts = get_possible_acts(observation, plr_color)
        _show_possible_acts(observation, possible_acts)

        action = -1
        while action < 0:
            _input = input("Input action : ")
            action_coord = [int(_i)-1 for _i in _input.split(',')]
            if action in possible_acts:
                action = coord2act(observation, action_coord)
            print(action)

        observation, _, done, _ = env.step(action)


def dqn_vs_random(dqn, verbose=False, env_name='Reversi8x8-v0'):
    """Execute dqn vs random."""
    assert gymenvs

    env = gym.make(env_name)
    plr_color = 0

    observation = env.reset()
    done = False
    while not done:
        if verbose:
            env.render()
        possible_acts = get_possible_acts(observation, plr_color)

        q_values = dqn.compute_q_values(
            observation.reshape((1,)+env.obs_space.shape))
        action = possible_acts[q_values[possible_acts].argmax()]

        observation, reward, done, _ = env.step(action)
    return env, reward
