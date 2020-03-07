"""Startup test."""
import gym
import random

import sys

sys.path.append('./')


def test_start_env():
    import revenius.gymenvs as gymenvs
    assert gymenvs
    env = gym.make('Reversi8x8-v0')

    env.reset()
    enables = env.possible_actions

    action = random.choice(enables)
    observation, reward, done, info = env.step(action)


def test_dqn():
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten
    from keras.optimizers import Adam
    from rl.agents.dqn import DQNAgent
    from rl.policy import EpsGreedyQPolicy
    from rl.memory import SequentialMemory

    import revenius.gymenvs as gymenvs
    assert gymenvs
    env = gym.make('Reversi8x8-v0')
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,)+env.obs_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation="linear"))

    memory = SequentialMemory(limit=5, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.001)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, gamma=0.99,
                   memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    history = dqn.fit(env, nb_steps=5, visualize=False, verbose=2)
    dqn.test(env, nb_episodes=1, visualize=True)

    print(history)
