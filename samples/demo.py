"""Demo samples."""
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import gym
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

sys.path.append('./')


def build_model(env):
    """Build model."""
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
    return model


def build_dqn(model, nb_actions):
    """Build DQN agent."""
    memory = SequentialMemory(limit=50000, window_length=1)

    policy = EpsGreedyQPolicy(eps=0.001)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, gamma=0.99,
                   memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


def main():
    """Exec main."""
    import revenius.gymenvs as gymenvs
    from revenius.execute import human_vs_dqn

    print(gymenvs)
    env = gym.make('Reversi8x8-v0')

    nb_actions = env.action_space.n
    model = build_model(env)
    dqn = build_dqn(model, nb_actions)

    dqn.fit(env, nb_steps=500, visualize=False, verbose=2)

    human_vs_dqn(dqn)


if __name__ == '__main__':
    main()
