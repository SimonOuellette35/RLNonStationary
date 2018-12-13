import random
import numpy as np
from collections import deque
import xgboost as xgb
import matplotlib.pyplot as plt

class RLAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.25
        self.model = None

    def remember(self, sar):
        self.memory.append(sar)

    def act(self, state, use_explo=True):
        if use_explo:
            if np.random.rand() <= self.epsilon or self.model is None:
                return random.randrange(self.action_size)

        act_values=[]
        for a in range(self.action_size):

            np_state = np.reshape(state, [1, self.state_size])
            np_action = np.reshape(a, [1, 1])

            input = np.concatenate((np_state, np_action), axis=1)

            input_xgb = xgb.DMatrix(input)
            action_Q = self.model.predict(input_xgb)

            act_values.append(action_Q)

        return np.argmax(act_values)  # returns optimal action

    def replay(self):
        # specify parameters via map
        max_depth = 5
        num_round = 2

        param = {'max_depth': max_depth, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}

        mem = np.array(self.memory)

        states = mem[:, :self.state_size]
        actions = np.reshape(mem[:, self.state_size], [-1, 1])
        rewards = mem[:, -1]

        # train an xgb model on current memory
        input_training = np.concatenate((states, actions), axis=1)

        training_data = xgb.DMatrix(input_training,
                                    label=rewards)

        self.model = xgb.train(param, training_data, num_round)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot_strategy(self):
        xgb.plot_tree(self.model)
        plt.show()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)