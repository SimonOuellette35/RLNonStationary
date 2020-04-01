import timevarying_dgp as tvdgp
from rl_agent import RLAgent
import numpy as np
import matplotlib.pyplot as plt


ESTIMATOR_WINDOW = 100
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

# =================================================== Part 3 =======================================================
print("Part 3: The solution -- simulation-based learning, with rolling estimate")
print("1. we generate several trajectories with our (theoretically perfect) generative model.")
print("2. we train the agent on those trajectories.")
print("3. we test that trained agent on new data from the same DGP: we show that its performance generalizes well (as predicted).")

print("Training the agent...")
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
            print(f"pct_progress = {pct_progress} %")
        else:
            print(f"pct_progress =  {pct_progress}% (current average P&L is {np.mean(training_pnls[-DELTA:])}")

agent.epsilon = 0.

print("Evaluating the agent's average performance...")
out_samplePnLs = []

# 3. test the agent on a series of stationary trajectories, show that it generalizes
for j in range(NUM_TEST_ITERATIONS):
    x_test, y_test = tvdgp.generateDGP(N)
    out_samplePnL = run_simulation(x_test, y_test, agent)
    out_samplePnLs.append(out_samplePnL)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

print(f"Average out-sample P&L across the tests: {np.mean(out_samplePnLs)} (standard deviation: {np.std(out_samplePnLs)})")

ax1.plot(training_pnls)
ax1.set_title("Part 3 - In-sample P&Ls")
ax2.plot(out_samplePnLs)
ax2.set_title("Part 3 - Out-of-sample P&Ls")
plt.show()

# Here we simulate model validation: we generate 2 trajectories, the first representing the
# historical realization that we would have used to produce and calibrate our generative model,
# and the second representing the future, unseen real data that we would expect our agent to
# generalize successfully to.
historicalX, historicalY = tvdgp.generateDGP(N)
futureX, futureY = tvdgp.generateDGP(N)

historicalPnL = run_simulation(historicalX, historicalY, agent)
futurePnL = run_simulation(futureX, futureY, agent)

print(f"P&L on historical realization: {historicalPnL}, P&L on future realization: {futurePnL}")