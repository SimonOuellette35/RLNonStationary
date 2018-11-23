import numpy as np
import matplotlib.pyplot as plt

def generateDGP(N=10000):
    # We have 2 cointegrated time series X and Y, related by some constant relationship, whose
    # successful estimation is necessary for optimal trading of assets X and Y.
    sigmaX = 0.05
    sigmaY = 0.05
    phi = 0.45
    intercept = 0.5
    slope = 0.6

    X = []
    Y = []
    epsilon = [0.]
    for t in range(N):
        if len(X) == 0:
            X.append(np.random.normal(10., sigmaX))
        else:
            X.append(X[-1] + np.random.normal(0., sigmaX))

        epsilon.append(phi * epsilon[t - 1] + np.random.normal(0., sigmaY))
        Y.append(intercept + slope * X[-1] + epsilon[-1])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, slope

X, Y, slope = generateDGP()
plt.plot(X)
plt.plot(Y)

plt.show()

# Visualize the spread
spread = Y - slope * X

plt.plot(spread)
plt.show()