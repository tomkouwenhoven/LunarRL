import numpy as np
import random
import gym
import os
import argparse

from matplotlib import pyplot as plt
from collections import deque
from tensorflow import keras
from tensorflow.python.keras.backend import conv2d
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D

env_name = "CarRacing-v0"
env = gym.make(env_name)

rewards = {}
replay_buffer = deque(maxlen=1000000)

epsilon = 0.5
alpha = 0.001
gamma = 0.99

epsilon_decay = 0.996

minibatch_size = 64

actions = [
    [-1.0, 0.0, 0.0], #left
    [-1.0, 0.3, 0.0], #soft left
    [ 1.0, 0.0, 0.0], #right
    [ 1.0, 0.3, 0.0], #soft right
    [ 0.0, 0.0, 1.0], #brake
    [ 0.0, 0.0, 0.5], #soft brake
    [ 0.0, 1.0, 0.0], #accelerate
    [ 0.0, 1.0, 0.8], #decelerate
    [ 0.0, 0.0, 0.0], #nothing
]

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(96,96,3)),
    keras.layers.Conv2D(32, kernel_size=8, padding='same', activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=5, padding='same', activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=3, padding='same', activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="linear"),
    keras.layers.Dense(9, activation="linear"),
])

optimizer = keras.optimizers.Adam(alpha)
model.compile(loss="mean_squared_error", optimizer=optimizer)
# model.summary()


def choose_action(state, mode):
    values = model.predict(state.reshape((1,96,96,3)))

    if mode == "play":
        idx = np.argmax(np.squeeze(values))
    else:
        if np.random.uniform(0,1) > epsilon:
            idx = np.argmax(np.squeeze(values))
            
        else:
            idx = np.random.randint(0,4)

    return actions[idx], idx


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


def train(num_episodes):
    # train agent
    for episode in range(num_episodes):
        cum_reward = 0
        state = env.reset()

        for step in range(1000):
            print(f"Step: {step:04}", end="\r")
            action, action_idx = choose_action(state, "train")

            next_state, reward, done, _ = env.step(action)

            experience = np.array([state, action_idx, reward, next_state, done], dtype='object')
            replay_buffer.append(experience)

            state = next_state

            cum_reward += reward

            if len(replay_buffer) > minibatch_size:
                samples = random.sample(replay_buffer, minibatch_size)
                replay(np.array(samples))

            # if episode % 5 == 0:
            #   env.render()
            
            rewards[episode] = cum_reward
            
            if done:
                break
             

        # pick random minibatch to train model.
        print(f"\nepisode #{episode:05} - epsilon: {epsilon} - reward: {cum_reward}")
    
    model_dir = os.path.join(os.getcwd(), "RL", "LunarRL", "models")
    figures_dir = os.path.join(os.getcwd(), "RL", "LunarRL", "figures")
    
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model.save(os.path.join(model_dir, "dqn_racecar.h5"))

    plt.plot(rewards.values())
    plt.savefig(os.path.join(figures_dir, "dqn_racecar.png"),
            dpi=300, 
            format='png')


def play():
    model_path = os.path.join(os.getcwd(), "RL", "LunarRL", "models", "dqn_racecar_v2.h5")

    if not os.path.exists(model_path):
        print("Model doesn't exist. Exiting...")
        return

    model.load_weights(model_path)

    # without training
    state = env.reset()

    for step in range(1000):
        action, action_idx = choose_action(state, "play")
        print(action)
        next_state, _, done, _ = env.step(action)
        state = next_state
        env.render()

        if done:
            break


def main(args=None):
    parser = argparse.ArgumentParser(description='Process settings for dqn lunarlander.')
    parser.add_argument('--mode', dest='mode',
                        type=str, default='train',
                        help='choose mode for agent, train or play')
    parser.add_argument('--episodes', dest='num_episodes',
                        type=int, default=100,
                        help='choose number of episodes to train or play')

    args = parser.parse_args()

    if args.mode == "train":
        train(args.num_episodes)
    elif args.mode == "play":
        play()


if __name__ == "__main__":
    main()