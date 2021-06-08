import time
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import gym
from gym.core import ObservationWrapper
env_name = "LunarLander-v2"
env = gym.make(env_name)

rewards = {}


num_episodes = 1
epsilon = 0.1

model = keras.Sequential()
model.add(keras.Input(shape=(8,)))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(4))

print(model.weights)

def choose_action(observation):
    preds = model(observation.reshape((1,8)))
    action = np.argmax(preds)
    print(action)
    return action


for episode in range(num_episodes):
    cum_reward = 0
    new_observation = env.reset()
    print(new_observation)
    for step in range(1000):

        action = choose_action(new_observation)

        new_observation, reward, done, info = env.step(action)

        cum_reward += reward

        arr = env.render(mode="rgb_array")
        if done:
          rewards[episode] = cum_reward
          break

# plt.plot(rewards.values())
# plt.show()
    # time.sleep(0.1)