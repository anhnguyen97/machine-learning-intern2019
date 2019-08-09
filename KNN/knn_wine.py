from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

winedata = datasets.load_wine()
wine_X = winedata.data
wine_Y = winedata.target
wine_features = winedata.feature_names
X_train, X_test, y_train, y_test = train_test_split(wine_X, wine_Y, test_size=50)

print("Data set:")
X0 = wine_X[wine_Y == 0, :]
print("Class 0: {}".format(len(X0)))
X1 = wine_X[wine_Y == 1, :]
print("Class 1: {}".format(len(X1)))
X2 = wine_X[wine_Y == 2, :]
print("Class 2: {}".format(len(X2)))

print("Number of class: {}".format(len(np.unique(wine_Y))))
print("Number of data set: {}".format(len(wine_X)))
print("- Test set: {}\n- Training set: {}".format(len(X_test), len(X_train)))
print("Training set:")
X0 = X_train[y_train == 0, :]
print("- Class 0: {}".format(len(X0)))
X1 = X_train[y_train == 1, :]
print("- Class 1: {}".format(len(X1)))
X2 = X_train[y_train == 2, :]
print("- Class 2: {}".format(len(X2)))


clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nResults with test data set with 1NN:")
print("- Prediction: {}".format(y_pred))
print("- Ground truth: {}".format(y_test))
print("Accuracy score with K=1: {}".format(100 * accuracy_score(y_test, y_pred)))


clf = neighbors.KNeighborsClassifier(n_neighbors=5, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy score with K=5: {}".format(100 * accuracy_score(y_test, y_pred)))

clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy score with K=10: {}".format(100 * accuracy_score(y_test, y_pred)))