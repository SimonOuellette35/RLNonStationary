import random
import numpy as np
from collections import deque
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt

class RLAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.5
        self.model = None
        self.MIN_VAL = 99.
        self.MAX_VAL = 101.
        self.D = int(((self.MAX_VAL - self.MIN_VAL) * 100) + 1)

    def remember(self, sar):
        self.memory.append(sar)

    def getMaxQ(self, state):

        if self.model is None:
            return 0.

        # In the Q-table we return the Q value of the best action (including flat, if both long and short are negative),
        # for the current state.
        action_slice = self.Qmus_estimates[:, self.state_value_to_index(state)]
        maxQ = np.max(action_slice, axis=0)

        if maxQ < 0.:
            return 0.       # we can just go flat, too
        else:
            return maxQ

    def act(self, state, use_explo=True):
        if use_explo:
           if np.random.rand() < self.epsilon or self.model is None:
               return random.randrange(self.action_size)

        action_slice = self.Qmus_estimates[:, self.state_value_to_index(state)]
        best_Q = np.max(action_slice)
        idx_best_action = np.argmax(action_slice)

        if best_Q < 0:
            return 1.
        elif idx_best_action == 0:
            return 0.
        else:
            return 2.

    def state_value_to_index(self, s):
        # map spread value to an index...
        idx = (s[0] - self.MIN_VAL) * 100

        return int(idx)

    def replay(self):

        mem = np.array(self.memory)

        states = mem[:, :self.state_size]
        actions = np.reshape(mem[:, self.state_size], [-1, 1])
        rewards = mem[:, -1]

        full_tensor = []
        s_short = []
        r_short = []
        for t in range(len(states)):

            idx = self.state_value_to_index(states[t])

            if actions[t] == 0:
                full_tensor.append(np.array([0, idx, rewards[t]]))
                s_short.append(idx)
                r_short.append(rewards[t])
            elif actions[t] == 2:
                full_tensor.append(np.array([1, idx, rewards[t]]))

        # plt.scatter(s_short, r_short)
        # plt.show()

        # qvalues = [N x 3]
        # 1 - action index (0 - short, 1 - long)
        # 2 - state index
        # 3 - reward
        qvalues = np.array(full_tensor)
        print "full_tensor shape = ", qvalues.shape


        # TODO: why does the more trained domain have a higher variance? Shouldn't increased certainty correlate with
        # lower variance???

        # TODO: why is it not learning anything above X = 125?

        with pm.Model() as self.model:

            def likelihood(Qmus, Qsds):

                def _logp(value):
                    idx0 = tt.cast(value[:, 0], dtype='int8')
                    idx1 = tt.cast(value[:, 1], dtype='int8')
                    return pm.Normal.dist(mu=Qmus[idx0, idx1], sd=np.exp(Qsds[idx0, idx1])).logp(value[:, 2])
                    #return pm.Normal.dist(mu=Qmus[idx0, idx1], sd=Qsds[idx0, idx1]).logp(value[:, 2])

                return _logp

            Qmus = pm.Normal('Qmus', mu=0., sd=1., shape=[2, self.D])
            Qsds = pm.Normal('Qsds', mu=-1., sd=.01, shape=[2, self.D])
            # Qmus = pm.Uniform('Qmus', lower=-1., upper=1., shape=[2, self.D])
            # Qsds = pm.Uniform('Qsds', lower=0.01, upper=0.2, shape=[2, self.D])

            pm.DensityDist('Qtable',
                           likelihood(Qmus, Qsds),
                           observed=qvalues)

            mean_field = pm.fit(n=8000, method='advi', obj_optimizer=pm.adam(learning_rate=.15))
            self.trace = mean_field.sample(5000)

            #self.trace = pm.sample(1000, tune=1000)

        self.Qmus_estimates = np.mean(self.trace['Qmus'], axis=0)
        self.Qsds_estimates = np.median(self.trace['Qsds'], axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(self.Qmus_estimates[0])
        ax1.set_title("Q-values for Short action")

        ax2.plot(self.Qsds_estimates[0])
        ax2.set_title("Certainties for Short action")

        plt.show()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
