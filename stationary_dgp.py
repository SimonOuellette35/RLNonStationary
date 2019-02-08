import numpy as np
import matplotlib.pyplot as plt

def generateDGP(N=10000):
    # We have 2 cointegrated time series X and Y, related by some constant relationship, whose
    # successful estimation is necessary for optimal trading of assets X and Y.
    sigmaX = 0.05
    sigmaEta = 0.1
    theta = 0.1
    mu = 100.

    X = []
    Y = []
    epsilon = [mu]
    for t in range(N):
        if len(X) == 0:
            X.append(np.random.normal(10., sigmaX))
        else:
            X.append(X[-1] + np.random.normal(0., sigmaX))

        epsilon.append(epsilon[-1] + theta * (mu - epsilon[-1]) + np.random.normal(0., sigmaEta))

        Y.append(X[-1] + epsilon[-1])

    X = np.array(X)
    Y = np.array(Y)

    discretized_X = np.round(X * 100.0) / 100.
    discretized_Y = np.round(Y * 100.0) / 100.

    return discretized_X, discretized_Y

# X, Y = generateDGP()
# plt.plot(X)
# plt.plot(Y)
#
# plt.show()
#
# # Visualize the spread
# spread = Y - X
#
# plt.plot(spread)
# plt.show()