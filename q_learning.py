from collections import deque
import numpy as np
import random
from agent import Agent
import os
import utils


# agent have (memory, env, ep_greedy)
class QLearning(Agent):
    def __init__(self):
        print('Create a QLearning Agent ------------')
        super().__init__()

    def train_model(self, episodes, max_step, training_file):
        self.q_table = np.zeros([self.state_size, self.action_size])
        c = 0
        for episode in range(episodes):
            state = self.env.reset()
            for step in range(max_step):
                # choose an action with ep-greedy(Q)
                action = self.select_action(state, episode)

                # takes action,  observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # Add a custom reward to reduce learning time
                if done:
                    # rewards (G = 10, H = -10)
                    self.q_table[next_state][:] = 2 * reward - 10

                # decrease reward every step
                reward = reward - 1
                self.update_q_table(state, action, next_state, reward)

                # update state
                state = next_state
                c += 1
                if done:
                    break
            #self.log_reward_of_episode(episode, c, step, reward)
            self.save_model(training_file)

    def all_Qs(self, state):
        return self.q_table[state]

    def new_q_value(self, next_state, reward):
        new_value = reward + self.gamma * np.amax(self.all_Qs(next_state))
        return new_value

    def update_q_table(self, state, action, next_state, reward):
        old_q_value = self.q_table[state][action]
        TD = self.new_q_value(next_state, reward) - old_q_value
        new_value = old_q_value + self.learning_rate * TD
        self.q_table[state][action] = new_value

    def save_model(self, file_name="my_qtable"):
        file_name = f'qlearning/{file_name}.txt'
        os.makedirs('qlearning', exist_ok=True)
        np.savetxt(file_name, self.q_table, delimiter=" ")

    def load_model(self, file_name="qlearning/my_qtable.txt"):
        self.q_table = np.zeros([self.state_size, self.action_size])
        if(os.path.isfile(file_name)):
            self.q_table = np.loadtxt(file_name)
