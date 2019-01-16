import stationary_dgp as sdgp
from rl_agent import RLAgent
import numpy as np
import matplotlib.pyplot as plt

N = 2000
REWARD_HORIZON = 100
DISCOUNT_DECAY = 0.98

def run_simulation(x, y, agent, train_agent=False):

    def current_reward(t, position):
        return ((y[t] - y[t - 1]) + (x[t - 1] - x[t])) * position

    def calculate_expected_reward_TD(t_idx, i, episodeSARs):
        expected_future_pnl = 0.

        discount_factor = DISCOUNT_DECAY

        position = episodeSARs[i][2] - 1
        for tau in range(1, REWARD_HORIZON + 1):
            if i+tau < len(episodeSARs):
                expected_future_pnl += discount_factor * ((y[t_idx + tau] - y[t_idx + tau - 1]) + (x[t_idx + tau - 1] - x[t_idx + tau])) * position

                discount_factor *= DISCOUNT_DECAY

        if i+tau < len(episodeSARs):
            final_state = [episodeSARs[i+tau][0], episodeSARs[i+tau][1]]
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
        state = [y[t] - x[t], current_position]
        action = agent.act(state)

        current_position = action - 1

        if train_agent and t >= 1:
            reward = current_reward(t, current_position)
            sar = [state[0], state[1], action, reward]
            episodeSARs.append(sar)

    if train_agent:
        for i in range(len(episodeSARs)):   # t = i + 1
            t_idx = i+1
            expected_future_pnl = calculate_expected_reward_TD(t_idx, i, episodeSARs)

            reward_label = episodeSARs[i][3] + expected_future_pnl
            tmpSAR = [episodeSARs[i][0], episodeSARs[i][1], episodeSARs[i][2], reward_label]

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
DELTA = 10
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
