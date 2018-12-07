import numpy as np
import matplotlib.pyplot as plt

# TODO: should the time-varying component be phi, or volatility, or both? fix beta to 1 for simplicity
def generateDGP(N=10000):
    sigmaX = 0.05
    sigmaEta = 0.1

    mu = 1.

    X = []
    Y = []
    epsilon = [mu]
    for t in range(N):
        if t % 200 == 0:
            theta = np.random.normal(0.1, 0.1)
            if theta < 0:
                theta = 0.

        if len(X) == 0:
            X.append(np.random.normal(10., sigmaX))
        else:
            X.append(X[-1] + np.random.normal(0., sigmaX))

        epsilon.append(epsilon[-1] + theta * (mu - epsilon[-1]) + np.random.normal(0., sigmaEta))

        Y.append(X[-1] + epsilon[-1])


    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# X, Y = generateDGP(2000)
#
# # plt.plot(X)
# # plt.plot(Y)
# #
# # plt.show()
#
# # Visualize the spread
# spread = Y - X
#
# plt.plot(spread)
# plt.show()