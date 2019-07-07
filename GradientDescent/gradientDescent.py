from builtins import print

import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

def main():

    # load data from file
    f = open('ex1data1.txt', "r")
    file = f.read()
    A = np.genfromtxt(StringIO(file), delimiter=',')
    f.close()
    X0 = np.array(A[:, 0]).reshape(97, 1)
    Y0 = np.array(A[:, 1]).reshape(97, 1)

    # Plotting the data from text file
    plt.figure()
    plt.plot(X0, Y0, 'rx')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.title("Plotting the data from data file")
    # plt.legend()   #chú thích biểu đồ
    plt.show()

    # tham so mo hinh
    m = len(X0)
    one = np.ones((97, 1))
    X = np.concatenate((one, X0), axis=1)
    alpha = 0.01
    theta = np.zeros((2, 1))
    print(theta.shape)
    iterators = 1500
    i = 0
    for i in range(iterators):
        deviation = X.dot(theta) - Y0
        theta[0] = theta[0] - (alpha/m)*sum(deviation)
        theta[1] = theta[1] - (alpha/m)*sum(deviation*X0)
        # print(theta)

    w_0 = theta[0]
    w_1 = theta[1]
    x0 = np.linspace(X.min(1), X.max(1), 2)
    y0 = w_0 + w_1 * x0

    # Drawing the fitting line
    X = A[:, 1]
    plt.plot(X0, Y0, 'rx')  # data
    plt.plot(x0, y0)  # the fitting line
    # plt.xlim(X.min()-1)
    plt.xlabel("Population of City in $10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.title("Plot gradient descent with 1 variable")
    plt.show()

def computeCost(X, Y, theta):
    prediction = X.dot(theta)
    return (1/(2*len(Y)))*sum(np.subtract(prediction, Y))

if __name__ == '__main__':
    main()