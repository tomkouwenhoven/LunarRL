import time
from matplotlib import pyplot as plt
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

import gym
from gym.core import ObservationWrapper
env_name = "LunarLander-v2"
env = gym.make(env_name)

rewards = {}
replay_buffer = deque(maxlen=500000)

num_episodes = 1000

epsilon = 0.5
alpha = 0.001
gamma = 0.99

epsilon_decay = 0.996

minibatch_size = 64

optimizer = tf.keras.optimizers.Adam(alpha)
model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(8,)))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(4, activation="linear"))

model.compile(loss="mean_squared_error", optimizer=optimizer)

def compute_loss(y, y_hat):
    return tf.square(y - y_hat)


model.compile(loss=compute_loss, optimizer=optimizer)

def choose_action(observation):
    preds = model(observation.reshape((1,8)))

    if np.random.uniform(0,1) > epsilon:
        action = np.argmax(preds)
    else:
        action = np.random.randint(0,3)
    return action

def train(samples):
    # x is 64 experiences with
    # {state, action, reward, next_state}

    states = np.array([i[0] for i in samples])
    actions = np.array([i[1] for i in samples])
    rewards = np.array([i[2] for i in samples])
    next_states = np.array([i[3] for i in samples])
    finishes = np.array([i[4] for i in samples])

    states = np.squeeze(states)
    next_states = np.squeeze(next_states)

    next_q_values = model.predict_on_batch(next_states)

    # calculate target and loss
    y = rewards + gamma * np.amax(next_q_values, axis=1) * (1 - finishes)

    indexes = np.array([i for i in range(minibatch_size)])
    targets = model.predict_on_batch(states)
    targets[[indexes], [actions]] = y

    model.fit(states, targets, epochs=1, verbose=0)


for episode in range(num_episodes):
    cum_reward = 0
    state = env.reset()

    for step in range(1000):
        action = choose_action(state)
        next_state, reward, done, info = env.step(action)

        experience = np.array([state, action, reward, next_state, done])
        replay_buffer.append(experience)

        state = next_state

        cum_reward += reward

        if episode % 100 == 0:
            arr = env.render(mode="rgb_array")

        if done:
          rewards[episode] = cum_reward
          break

    if epsilon > 0.01:
        epsilon *= epsilon_decay

    # pick random minibatch to train model.
    print(f"\nepisode #{episode:05} - epsilon: {epsilon} - reward: {cum_reward}")

    if len(replay_buffer) > minibatch_size:
        train(np.array(random.sample(replay_buffer, minibatch_size)))



plt.plot(rewards.values())
plt.show()
    # time.sleep(0.1)