import numpy as np
from io import  StringIO
import matplotlib.pyplot as plt

def main():
    f = open("ex1data1.txt", "r")
    file = f.read()
    A = np.genfromtxt(StringIO(file),delimiter=',')
    f.close()
    x = A[:, 0]
    y = A[:, 1]

    # Plotting the data from text file
    plt.figure()
    plt.plot(x, y, 'rx')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.title("My plot")
    # plt.legend()   chú thích biểu đồ
    plt.show()

    # Calculating weights of the fitting line
    A = np.c_[ np.ones(len(x)), A ]
    theta = np.ones(2).transpose()
    X = A[:, 0:2]
    Y = A[:, 2]
    w = ((np.linalg.pinv(X.transpose().dot(X))).dot(X.transpose())).dot(Y)
    print('w = ', w)

    w_0 = w[0]
    w_1 = w[1]
    x0 = np.linspace(X.min(1), X.max(1), 2)
    y0 = w_0 + w_1 * x0

    # Drawing the fitting line
    X = A[:, 1]
    plt.plot(X, Y, 'rx')  # data
    plt.plot(x0, y0)  # the fitting line
    # plt.xlim(X.min()-1)
    plt.xlabel("Population of City in $10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()


def cost(A, theta):
    X = A[:, 0:2]
    Y = A[:, 2]
    m = len(Y)
    predection = 1/(2*m)*sum((X.dot(theta) - Y) ** 2)
    return predection

if __name__=="__main__":
    main()