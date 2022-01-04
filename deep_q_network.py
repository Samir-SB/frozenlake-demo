from collections import deque
import numpy as np
import random
from agent import Agent
import utils
import os


class DQN(Agent):
    def __init__(self):
        # super().method(arg)
        super(DQN, self).__init__()
        self.initialize_networks()
        self.syncronize_networks()
        self.set_replay_prameters()

    def train_model(self, episodes, max_step, training_file):
        self.memory = deque(maxlen=self.memory_maxlen)
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
                    reward = 20 * reward - 10
                if next_state == state:
                    reward = -2

                # Store transition in the experience replay memory.
                self.memory.append((state, next_state, reward, action, done))

                # update state
                state = next_state
                c += 1
                if done:
                    break

            self.log_reward_of_episode(episode, c, step, reward)
            if len(self.memory) < self.enough_experiences:
                continue

            self.replay_experiences()
            accuracy = self.evaluate()
            print('accuracy = ', accuracy)
            if (accuracy > self.traget_accuracy):
                self.save_model(f'converged/converged-at-{episode}')
                break

            if episode % 15 == 0:
                self.syncronize_networks()
                self.save_model(training_file)

    def replay_experiences(self):
        minibatch = random.sample(self.memory, self.batch_size)
        inputs = []
        targets = []
        for state, next_state, reward, action, done in minibatch:
            # all predict Qs equals target_Qs
            # except Q of selected action equals update_q
            # Use Target netwok to pridect updated Q
            x, y = self.generate_x_and_y(state)
            y[action] = self.updated_q(next_state, reward, done)
            inputs.append(x)
            targets.append(y)
        X = np.stack(inputs, axis=0)
        Y = np.stack(targets, axis=0)
        self.main_network.fit(X, Y, epochs=1)
        print('fit------')

    def generate_x_and_y(self, state):
        reshaped_state = np.identity(self.state_size)[state: state + 1]
        x = reshaped_state[0]
        y = self.main_network.predict(reshaped_state)[0]
        return (x, y)

    def all_Qs(self, state, use_target=False):
        reshaped_state = np.identity(self.state_size)[state: state + 1]
        if use_target:
            return self.target_network.predict(reshaped_state)
        return self.main_network.predict(reshaped_state)

    def updated_q(self, next_state, reward, done):
        if done:
            return reward
        # apply bellman equation
        return reward + self.gamma * np.amax(self.all_Qs(next_state, True))

    def set_replay_prameters(self, memory_maxlen=100000, enough_experiences=2000, batch_size=32):
        self.memory_maxlen = memory_maxlen
        self.enough_experiences = enough_experiences
        self.batch_size = batch_size

    def initialize_networks(self):
        self.main_network = utils.init_network(
            self.state_size, self.action_size, self.learning_rate)
        self.target_network = utils.init_network(
            self.state_size, self.action_size, self.learning_rate)

    def syncronize_networks(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def save_model(self, file_name="my_checkpoint"):
        file_name = f'dqn/{file_name}'
        self.main_network.save_weights(file_name)
        self.env.render()
        self.print_policy()

    def load_model(self, file_name="dqn/my_checkpoint"):
        if(os.path.isfile(f'{file_name}.index')):
            print('load module from: ', file_name)
            self.main_network.load_weights(file_name)
