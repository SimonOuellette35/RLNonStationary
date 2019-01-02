import timevarying_dgp as tvdgp
from rl_agent import RLAgent
import numpy as np
import matplotlib.pyplot as plt

N = 2000
REWARD_HORIZON = 100
DISCOUNT_DECAY = 0.98

def run_simulation(x, y, agent, train_agent=False):

    def calculate_reward(t, position):
        forward_pnl = 0
        discount_factor = 1.
        for tau in range(1, REWARD_HORIZON+1):
            if position == 1:
                forward_pnl += discount_factor * ((y[t+tau] - y[t+tau- 1]) + (x[t+tau - 1] - x[t+tau]))
            elif position == -1:
                forward_pnl += discount_factor * ((y[t+tau - 1] - y[t+tau]) + (x[t+tau] - x[t+tau - 1]))
            else:
                return 0.0

            discount_factor *= DISCOUNT_DECAY

        return forward_pnl

    current_position = 0
    current_pnl = 0
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

        if train_agent:
            reward = calculate_reward(t, current_position)
            sar = [state[0], state[1], action, reward]
            agent.remember(sar)

    return current_pnl

NUM_TRAINING_ITERATIONS = 100
NUM_TEST_ITERATIONS = 100

# =================================================== Part 2 =======================================================
print "Part 2: Optimizing on non-stationary historical data..."
print "1. we get a specific historical realization of data from our time-varying DGP."
print "2. we train the agent on that trajectory."
print "3. we test that trained agent on new data from the same DGP: we show that its performance does NOT generalize well (as predicted)."

in_samplePnLs = []
out_samplePnLs = []

# 1. generate a non-stationary "historical" trajectory
x, y = tvdgp.generateDGP(N)
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

print "Agent epsilon after training: ", agent.epsilon
agent.epsilon = 0.

in_samplePnL = run_simulation(x, y, agent)
print "Average P&L after training: ", in_samplePnL

print "Testing the agent..."
# 3. test the agent on a series of stationary trajectories, show that it generalizes
for j in range(NUM_TEST_ITERATIONS):
    x_test, y_test = tvdgp.generateDGP(N)
    out_samplePnL = run_simulation(x_test, y_test, agent)
    out_samplePnLs.append(out_samplePnL)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

ax1.plot(training_pnls)
ax1.set_title("Part 2 - In-sample P&Ls")
ax2.plot(out_samplePnLs)
ax2.set_title("Part 2 - Out-of-sample P&Ls")
plt.show()

print "Average out-sample P&L across the tests: ", np.mean(out_samplePnLs)
