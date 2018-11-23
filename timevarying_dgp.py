import numpy as np
import matplotlib.pyplot as plt

def generateDGP(N=10000):
    # We have 2 cointegrated time series X and Y, related by some non-stationary, stochastic relationship, whose
    # successful estimation is necessary for optimal trading of assets X and Y.
    sigmaX = 0.05
    sigmaY = 0.05
    phi = 0.45
    intercept = 0.5

    slope_theta = 0.005
    slope_mu = 0.6
    slope_sigma = 0.005

    X = []
    Y = []
    slope = [slope_mu]
    epsilon = [0.]
    for t in range(N):
        if len(X) == 0:
            X.append(np.random.normal(10., sigmaX))
        else:
            X.append(X[-1] + np.random.normal(0., sigmaX))

        epsilon.append(phi * epsilon[t - 1] + np.random.normal(0., sigmaY))
        Y.append(intercept + slope[-1] * X[-1] + epsilon[-1])

        slope.append(slope[-1] + slope_theta * (slope_mu - slope[-1]) + np.random.normal(0., slope_sigma))

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, slope_mu

X, Y, slope_mu = generateDGP()
plt.plot(X)
plt.plot(Y)

plt.show()

# Visualize the spread
spread = Y - slope_mu * X

plt.plot(spread)
plt.show()