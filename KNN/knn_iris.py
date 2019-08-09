from sklearn import datasets, neighbors
import numpy as np
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
iris_label = iris.target_names
# print(iris)
print("Number of data: ", len(iris_X))
print("Number of label: ", len(iris_label))

X0 = iris_X[iris_Y == 0, :]
# print('\nSamples from class {}:\n {}'.format(iris_label[0], X0))

X1 = iris_X[iris_Y == 1, :]
# print('\nSamples from class {}:\n {}'.format(iris_label[1], X1))

X2 = iris_X[iris_Y == 2, :]
# print('\nSamples from class {}:\n {}'.format(iris_label[2], X2))

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=50)

print("Training size: {}".format(len(y_train)))
print("Test size    : {}\n".format(len(y_test)))

# KNN with k = 1
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# y_prob = clf.predict_proba(X_test)

print("Results of 20 test data point:")
print("Predicted labels: {}".format(y_pred[10:30]))
print("Ground truth: {}\n".format(y_test[10:30]))

from sklearn.metrics import accuracy_score

print("Accuracy score of KNN with K=1: {} %".format(100 * accuracy_score(y_test, y_pred)))

# KNN with k=5
clf = neighbors.KNeighborsClassifier(n_neighbors=5, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy score of KNN with K=5: {} %".format(100 * accuracy_score(y_test, y_pred)))  # KNN with k=10

# KNN with K = 10
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy score of KNN with K=10: {} %".format(100 * accuracy_score(y_test, y_pred)))

# KNN with weights for neighbor
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy score of 10NN (1/distance weights): {} %".format(100 * accuracy_score(y_test, y_pred)))
