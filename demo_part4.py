import timevarying_dgp as tvdgp
from rl_agent import RLAgent
import numpy as np
import matplotlib.pyplot as plt

N = 2000
REWARD_HORIZON = 100
DISCOUNT_DECAY = 0.98
ESTIMATOR_WINDOW = 100

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
    spreads = []
    for t in range(len(x) - REWARD_HORIZON):

        # calculate P&L accrued from whatever position we were in at last time step.
        if current_position == 1:
            current_pnl += (y[t] - y[t-1]) + (x[t-1] - x[t])
        elif current_position == -1:
            current_pnl += (y[t-1] - y[t]) + (x[t] - x[t-1])

        spread = y[t] - x[t]
        spreads.append(spread)
        start_idx = t - ESTIMATOR_WINDOW
        if start_idx < 0:
            start_idx = 0

        mu_estimate = np.mean(spreads[start_idx:t+1])

        real_spread = spread - mu_estimate

        # summarize the current state of things, and query the agent for our next action
        state = [real_spread, current_position]
        action = agent.act(state)

        current_position = action - 1

        if train_agent:
            reward = calculate_reward(t, current_position)
            sar = [state[0], state[1], action, reward]
            agent.remember(sar)

    return current_pnl

NUM_TRAINING_ITERATIONS = 100
NUM_TEST_ITERATIONS = 100

# =================================================== Part 4 =======================================================
print "Part 4: The solution -- simulation-based learning, with rolling estimate"
print "1. we generate several trajectories with our (theoretically perfect) generative model."
print "2. we train the agent on those trajectories."
print "3. we test that trained agent on new data from the same DGP: we show that its performance generalizes well (as predicted)."

print "Training the agent..."
agent = RLAgent(2, 3)
training_pnls = []
DELTA = 20
for j in range(NUM_TRAINING_ITERATIONS):
    # generate a non-stationary trajectory simulation
    x, y = tvdgp.generateDGP(N)

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

print "Evaluating the agent's average performance..."
out_samplePnLs = []

# 3. test the agent on a series of stationary trajectories, show that it generalizes
for j in range(NUM_TEST_ITERATIONS):
    x_test, y_test = tvdgp.generateDGP(N)
    out_samplePnL = run_simulation(x_test, y_test, agent)
    out_samplePnLs.append(out_samplePnL)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

print "Average out-sample P&L across the tests: %s (standard deviation: %s)" % (
    np.mean(out_samplePnLs),
    np.std(out_samplePnLs)
)

ax1.plot(training_pnls)
ax1.set_title("Part 4 - In-sample P&Ls")
ax2.plot(out_samplePnLs)
ax2.set_title("Part 4 - Out-of-sample P&Ls")
plt.show()

# Here we simulate model validation: we generate 2 trajectories, the first representing the
# historical realization that we would have used to produce and calibrate our generative model,
# and the second representing the future, unseen real data that we would expect our agent to
# generalize successfully to.
historicalX, historicalY = tvdgp.generateDGP(N)
futureX, futureY = tvdgp.generateDGP(N)

historicalPnL = run_simulation(historicalX, historicalY, agent)
futurePnL = run_simulation(futureX, futureY, agent)

print "P&L on historical realization: %s, P&L on future realization: %s" % (
    historicalPnL,
    futurePnL
)