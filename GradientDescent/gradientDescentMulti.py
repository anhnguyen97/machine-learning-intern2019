from builtins import print

import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from setuptools._vendor.six import print_


def main():
    # load data from data file
    f = open('ex1data2.txt', 'r')
    file = f.read()
    A = np.genfromtxt(StringIO(file), delimiter=',')
    f.close()

    X = A[:, 0:2]
    Y = A[:, 2]
    m = len(Y)

    [X_std, X_mean, X_nor] = featureNormalize(X)
    [Y_std, Y_mean, Y_nor] = featureNormalize(Y)
    one = np.ones((X.shape[0], 1))
    X_nor = np.concatenate((one, X_nor), axis=1)
    X_one = np.concatenate((one, np.reshape(X, (m, 2))), axis=1)
    Y_nor = np.reshape(Y, (m, 1))

    alpha = 0.001  # learning rate
    theta = np.zeros((3, 1))  # parameter
    iterators = 150  # number of loop to converge
    J = np.reshape(np.zeros((iterators, 1)), (iterators, 1))
    i = 1
    for i in range(iterators):
        pred = X_nor.dot(theta)
        updates = np.transpose(X_nor).dot(pred - Y_nor)
        # theta[0] = theta[0] - (alpha / m) * np.sum(pred * X_nor[:, 0])
        # theta[1] = theta[1] - (alpha / m) * np.sum(pred * X_nor[:, 1])
        # theta[2] = theta[2] - (alpha / m) * np.sum(pred * X_nor[:, 2])
        theta = theta - alpha * (1 / m) * updates
        J[i] = computeCost(X_nor, theta, Y_nor)
        if (J[i] - J[i - 1] < 0.001) and i>1:
            print(i)
            break

    print_(np.around(theta, decimals=5))
    house = [1, 4478, 5]
    # normalize:
    house[1] = (house[1] - X_mean[0]) / X_std[0]
    house[2] = (house[2] - X_mean[1]) / X_std[1]
    nomalized_cost = np.dot(house, theta)
    cost = nomalized_cost * Y_std + Y_mean
    # cost = np.dot(house, theta)
    print("Gia nha cua can nha co dien tich ", house[1], " va so phong ", house[2], " la: ", cost)
    print(house, nomalized_cost)
    # print(X_std, X_mean, Y_std, Y_mean)


def featureNormalize(X):
    std = np.std(X, axis=0)
    mean = np.mean(X, axis=0)
    X_nor = np.divide(np.subtract(X, mean), std)
    return std, mean, X_nor


def computeCost(X, theta, Y):
    theta = np.reshape(theta, (3, 1))
    Y = np.reshape(Y, (len(Y), 1))
    return (1 / (2 * len(Y)) * sum((X.dot(theta) - Y) ** 2))


if __name__ == '__main__':
    main()
