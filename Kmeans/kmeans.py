import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random


def main():
    X = np.array([[2, 8], [2, 5], [1, 2], [5, 8], [7, 3], [6, 4], [8, 4], [4, 7]])
    # K: number of centers / centroids
    K = 3

    visualize(X)

    centers = [kmeans_init_centers(X, K)]
    # centers = np.array([[X[index] for index in centers_index]])

    while True:
        # gán nhãn cho các điểm dữ liệu
        labels = kmeans_assign_label(X, centers[-1])
        new_centers = kmeans_update_centers(X, K, labels)
        # print(new_centers, '\n', centers)
        if set([tuple(a) for a in centers[-1]]) == set([tuple(a) for a in new_centers]):
            break
        centers.append(new_centers)

    visualize_cluster(X, labels, centers[-1])
    print(centers[-1])


def visualize_cluster(X, labels, centers):
    X0 = X[labels == 0, :]
    X1 = X[labels == 1, :]
    X2 = X[labels == 2, :]
    # print(X0, Y0)

    plt.figure()
    plt.title('Clustering using Kmeans')
    plt.plot(X0[:, 0], X0[:, 1], 'bo', Markersize=4, alpha=0.8)
    plt.plot(X1[:, 0], X1[:, 1], 'rs', Markersize=4, alpha=0.8)
    plt.plot(X2[:, 0], X2[:, 1], 'g^', Markersize=4, alpha=0.8)

    # Visualize Centers
    plt.plot(centers[0][0], centers[0][1], 'yx', Markersize=4, alpha=0.8)
    plt.plot(centers[1][0], centers[1][1], 'yx', Markersize=4, alpha=0.8)
    plt.plot(centers[2][0], centers[2][1], 'yx', Markersize=4, alpha=0.8)

    plt.axis('equal')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.plot()
    plt.show()


def visualize(X):
    X0 = X[:, 0]
    Y0 = X[:, 1]
    # print(X0, Y0)

    plt.figure()
    plt.title('Clustering using Kmeans')
    plt.plot(X0, Y0, 'bo', Markersize=4, alpha=0.8)
    plt.axis('equal')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.plot()
    plt.show()


def kmeans_init_centers(X, K):
    # random 3 centroids
    # return X[random.sample(range(8), 3)]
    return X[np.random.choice(X.shape[0], K, replace=False)]


# gán nhãn cho các điểm dữ liệu khi biết các center points
def kmeans_assign_label(X, centers):
    # distance between points in data_set
    D = distance.cdist(X, centers, "euclidean")

    # return label(0,1,2) tuong ung center cho cac point in X
    return D.argmin(axis=1)


# update center mới sau khi cập nhật label cho các điểm dữ liệu
def kmeans_update_centers(X, K, labels):
    centers = np.zeros((K, X.shape[1]))
    # print(labels)
    for k in range(K):
        Xk = X[labels == k, :]
        # print(Xk)
        centers[k, :] = np.mean(Xk, axis=0)
        # print(centers)
    return centers


if __name__ == '__main__':
    main()
