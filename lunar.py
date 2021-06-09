import numpy as np
import random
import gym
import os
import argparse

from matplotlib import pyplot as plt
from collections import deque
from tensorflow import keras

env_name = "LunarLander-v2"
env = gym.make(env_name)

rewards = {}
replay_buffer = deque(maxlen=1000000)

num_episodes = 1000

epsilon = 0.5
alpha = 0.001
gamma = 0.99

epsilon_decay = 0.996

minibatch_size = 64

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(8,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(4, activation="linear"),
])
optimizer = keras.optimizers.Adam(alpha)
model.compile(loss="mean_squared_error", optimizer=optimizer)


def choose_action(state, mode):
    values = model.predict(state.reshape((1,8)))

    if mode == "play":
        action = np.argmax(values)
    else:
        if np.random.uniform(0,1) > epsilon:
            action = np.argmax(values)
        else:
            action = np.random.randint(0,3)
    return action


def replay(samples):
    states = np.array([i[0] for i in samples])
    actions = np.array([i[1] for i in samples])
    rewards = np.array([i[2] for i in samples])
    next_states = np.array([i[3] for i in samples])
    finishes = np.array([i[4] for i in samples])

    states = np.squeeze(states)
    next_states = np.squeeze(next_states)

    next_q_values = model.predict_on_batch(next_states)

    # calculate target
    y = rewards + gamma * np.amax(next_q_values, axis=1) * (1 - finishes)

    indexes = np.array([i for i in range(minibatch_size)])
    targets = model.predict_on_batch(states)
    targets[[indexes], [actions]] = y

    model.fit(states, targets, epochs=1, verbose=0)

    global epsilon
    if epsilon > 0.01:
        epsilon *= epsilon_decay


def train():
    # train agent
    for episode in range(num_episodes):
        cum_reward = 0
        state = env.reset()

        for step in range(1000):
            action = choose_action(state, "train")
            next_state, reward, done, _ = env.step(action)

            experience = np.array([state, action, reward, next_state, done])
            replay_buffer.append(experience)

            state = next_state

            cum_reward += reward

            if len(replay_buffer) > minibatch_size:
                samples = random.sample(replay_buffer, minibatch_size)
                replay(np.array(samples))

            if episode % 5 == 0:
                env.render(mode="rgb_array")

            if done:
                rewards[episode] = cum_reward
                break


        # pick random minibatch to train model.
        print(f"\nepisode #{episode:05} - epsilon: {epsilon} - reward: {cum_reward}")

    if os.path.exists("models"):
        os.mkdir("models")

    model.save("models/dqn_lunarlander.h5")

    plt.plot(rewards.values())
    plt.show()


def play():
    if not os.path.exists("models/dqn_lunarlander.h5"):
        print("Model doesn't exist. Exiting...")
        return

    model.load_weights("models/dqn_lunarlander.h5")

    # without training
    state = env.reset()

    for step in range(1000):
        action = choose_action(state, "play")

        next_state, _, done, _ = env.step(action)
        state = next_state
        env.render(mode="rgb_array")

        if done:
            break


def main(args=None):
    parser = argparse.ArgumentParser(description='Process settings for dqn lunarlander.')
    parser.add_argument('--mode', dest='mode',
                        type=str, default=train,
                        help='choose mode for agent, train or play')

    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "play":
        play()


if __name__ == "__main__":
    main()