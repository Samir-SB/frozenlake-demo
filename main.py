from deep_q_network import DQN
import os

import time
from q_learning import QLearning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def model_is_converged(file_name):
    agent.load_model(file_name)
    accuracy = agent.evaluate()
    return (accuracy > .99)


def train_model_with_dqn(start_file=''):
    # only for dqn
    agent.set_replay_prameters(5000, 1000, 32)
    while(not model_is_converged(start_file)):
        print('need some optimization')
        agent.train_model(276, 100, training_file)
        agent.save_model(backup_file)

    print('model loaded is converged-------------')
    agent.save_model(converged_file)


def train_model_with_qlearning(start_file=''):
    while(not model_is_converged(start_file)):
        agent.print_policy()
        print('need some optimization')
        agent.train_model(5000, 100, training_file)
        agent.save_model(backup_file)
        time.sleep(.5)

    print('model loaded is converged-------------')
    agent.save_model(converged_file)
    print(agent.q_table)

# Some examples of converged dqn models.
# start_file = 'converged/converged-at-340'
# start_file = 'converged/converged-at-1127'
# start_file = 'converged/v1-165/my_checkpoint'


# qlearing table
start_file = 'qlearning/backup.txt'

backup_file = 'backup'
training_file = 'training'
converged_file = 'converged-3'
agent = QLearning()  # DQN()
# If you want custom parameters
# agent.set_ep_greedy_parameters(.4)
# agent.set_hyperparameters(.002)
# train_model_with_dqn()
train_model_with_qlearning(start_file)
