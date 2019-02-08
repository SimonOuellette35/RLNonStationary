import stationary_dgp as sdgp
from rl_agent import RLAgent
import numpy as np
import matplotlib.pyplot as plt

N = 2000
REWARD_HORIZON = 100
DISCOUNT_DECAY = 0.95

def run_simulation(x, y, agent, train_agent=False):

    def current_reward(t, position):
        return ((y[t+1] - y[t]) + (x[t] - x[t+1])) * position

    def calculate_expected_reward_TD(t, episodeSARs):
        expected_future_pnl = 0.

        discount_factor = DISCOUNT_DECAY

        position = episodeSARs[t][2] - 1
        for tau in range(2, REWARD_HORIZON + 1):
            if t+tau < len(episodeSARs):
                expected_future_pnl += discount_factor * ((y[t+tau] - y[t+tau-1]) + (x[t+tau-1] - x[t+tau])) * position

                discount_factor *= DISCOUNT_DECAY

        if t+tau < len(episodeSARs):
            final_state = [episodeSARs[t+tau][0], episodeSARs[t+tau][1]]
            Q = agent.getMaxQ(final_state)

            return expected_future_pnl + DISCOUNT_DECAY * Q
        else:
            return expected_future_pnl

    current_position = 0
    current_pnl = 0
    episodeSARs = []
    for t in range(len(x) - REWARD_HORIZON):

        # calculate P&L accrued from whatever position we were in at last time step.
        if current_position == 1:
            current_pnl += (y[t] - y[t-1]) + (x[t-1] - x[t])
        elif current_position == -1:
            current_pnl += (y[t-1] - y[t]) + (x[t] - x[t-1])

        # summarize the current state of things, and query the agent for our next action
        state = [y[t] - x[t]]
        action = agent.act(state)

        current_position = action - 1

        if train_agent:
            reward = current_reward(t, current_position)
            sar = [state[0], state[1], action, reward]
            episodeSARs.append(sar)

    if train_agent:
        for t in range(len(episodeSARs)):
            expected_future_pnl = calculate_expected_reward_TD(t, episodeSARs)

            reward_label = episodeSARs[t][3] + expected_future_pnl
            tmpSAR = [episodeSARs[t][0], episodeSARs[t][1], episodeSARs[t][2], reward_label]

            agent.remember(tmpSAR)

    return current_pnl

NUM_TRAINING_ITERATIONS = 100
NUM_TEST_ITERATIONS = 100

# =================================================== Part 1 =======================================================
print "Part 1: Optimizing on purely stationary time series"
print "1. we get a specific historical realization of data from our stationary DGP."
print "2. we train the agent on that trajectory."
print "3. we test that trained agent on new data from the same DGP: we show that its performance generalizes well (as predicted)."

in_samplePnLs = []
out_samplePnLs = []

# 1. generate a stationary "historical" trajectory
x, y = sdgp.generateDGP(N)
spread = y - x
plt.plot(spread)
plt.show()

print "Training the agent..."
# 2. train the agent on that trajectory, show that it learned some optimum
agent = RLAgent(2, 3)
training_pnls = []
DELTA = 20
for j in range(NUM_TRAINING_ITERATIONS):
    training_pnl = run_simulation(x, y, agent, True)
    training_pnls.append(training_pnl)
    if j % DELTA == 0:
        agent.replay()

        pct_progress = (float(j) / float(NUM_TRAINING_ITERATIONS)) * 100.0
        if j == 0:
            print "pct_progress = %s %%" % (pct_progress)
        else:
            print "pct_progress = %s %% (current average P&L is %s)" % (pct_progress, np.mean(training_pnls[-DELTA:]))

agent.epsilon = 0.

training_pnl = run_simulation(x, y, agent, False)
print "Training P&L = ", training_pnl

print "Testing the agent..."
# 3. test the agent on a series of stationary trajectories, show that it generalizes
for j in range(NUM_TEST_ITERATIONS):
    x_test, y_test = sdgp.generateDGP(N)
    out_samplePnL = run_simulation(x_test, y_test, agent)
    out_samplePnLs.append(out_samplePnL)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

ax1.plot(training_pnls)
ax1.set_title("Part 1 - In-sample P&Ls")
ax2.plot(out_samplePnLs)
ax2.set_title("Part 1 - Out-of-sample P&Ls")
plt.show()

print "Average out-sample P&L across the tests: ", np.mean(out_samplePnLs)
