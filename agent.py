from collections import deque
import random
import numpy as np
#from tensorflow.python.keras.models import clone_model
import tensorflow as tf
#from main import reshape_state
import utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Agent:
    def __init__(self, lr=0.001, gamma=.99, start=1, end=0.05, decay=.99, test_episodes=25, min_accuracy=.85):
        print('Create & initialize an agent ------------')
        env = utils.createEnvironment()
        self.set_environemnt(env)
        self.set_hyperparameters(lr, gamma)
        self.set_ep_greedy_parameters(start, end, decay)
        self.set_evaluation_parameters(test_episodes, min_accuracy)

    def set_hyperparameters(self, lr=0.001, gamma=0.99):
        self.learning_rate = lr
        self.gamma = gamma

    def set_evaluation_parameters(self, test_episodes, min_accuracy):
        self.test_episodes = test_episodes
        self.min_accuracy = min_accuracy

    def set_environemnt(self, env):
        self.env = env
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n

    def set_ep_greedy_parameters(self, start=1, end=.05, decay=.99):
        self.start = start
        self.end = end
        self.decay = decay

    def all_Qs(self, state):
        pass

    def predict(self, state):
        return np.argmax(self.all_Qs(state))

    def get_epsilon(self, step=0):
        return max(self.end, self.start * (self.decay**step))

    def select_action(self, state, step):
        epsilon = self.get_epsilon(step)
        random_selection = (np.random.rand() <= epsilon)
        return np.random.randint(0, self.action_size) if(random_selection) \
            else self.predict(state)

    def evaluate(self):
        total_reward = 0
        for episode in range(self.test_episodes):
            env = utils.random_env()
            state = env.reset()
            episode_reward = 0
            step = 0
            while True:
                step = step + 1
                action = self.predict(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                if(done or (next_state == state) or (step > self.state_size)):
                    total_reward = total_reward + reward
                    break
                state = next_state
        env.render()
        self.print_policy(4)
        accuracy = total_reward / self.test_episodes
        print(accuracy)
        return (accuracy)

    def print_policy(self, ROW_SIZE=4):
        display_actions = [' ⬅ ', ' ⬇ ', ' ➡ ', ' ⬆ ']
        for i in range(self.state_size):
            if 0 == (i % ROW_SIZE):
                print()
            print(display_actions[self.predict(i)], end='')
        print()

    def log_reward_of_episode(self, episode, global_step, step, reward):
        print(episode, end=': ')
        utils.print_line(global_step, self.get_epsilon(episode), step, reward)
