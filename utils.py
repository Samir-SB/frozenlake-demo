# from collections import deque
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gym


def get_custom_map(x=0, y=0):
    custom_map = [
        'FFFF',
        'FFFH',
        'FFFF',
        'HFFG'
    ]
    # x = 0 #np.random.randint(0, 4)
    # y = np.random.randint(0, 4)
    list_map = list(custom_map[x])
    if(list_map[y] == 'F'):
        list_map[y] = 'S'
    custom_map[x] = ''.join(list_map)
    return custom_map


def random_env():
    x = np.random.randint(0, 4)
    y = np.random.randint(0, 4)
    random_map = get_custom_map(x, y)
    env = createEnvironment(random_map)
    return env


def createEnvironment(custom_map=get_custom_map()):
    env_version = 'FrozenLake-v1' # 'FrozenLake-v0'
    env = gym.make(env_version, desc=custom_map, is_slippery=False)
    return env


def init_network(input_dim, output_dim, learning_rate=0.001, name='myModel'):
    hidden_dim = (input_dim + output_dim) // 2
    model = Sequential(name=name)
    model.add(Dense(hidden_dim, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    # model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model


def print_line(global_step, epsilon, step, reward):
    print(
        f'global_step {global_step}: epsilon = {epsilon} - step = {step} - reward = {reward}')


def to_csv(filename, data):
    pd.DataFrame(data).to_csv(filename, index=False)


def load_csv(filename):
    # Create a dataframe object from the csv file
    dfObj = pd.read_csv(filename, delimiter=',')
    # Create a list of tuples for Dataframe rows using list comprehension
    data = [tuple(row) for row in dfObj.values]
    return data
