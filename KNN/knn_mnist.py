from sklearn import neighbors
import numpy as np
from mnist import MNIST
from sklearn.metrics import accuracy_score
import time

mndata = MNIST('../MNIST/')
mndata.load_testing()
mndata.load_training()

X_train = mndata.train_images
X_test = mndata.test_images
y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)

start_time = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
print("Accuracy of 1NN for MNIST: {} %".format(100 * accuracy_score(y_test, y_pred)))
print("Running time: {} (s)".format(end_time - start_time))


start_time = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors=5, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
print("Accuracy of 5NN for MNIST: {} %".format(100 * accuracy_score(y_test, y_pred)))
print("Running time: {} (s)".format(end_time - start_time))

