from itertools import count
import numpy as np
from numpy import size
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import *
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from pyvi import ViTokenizer


def load_data(filename):
    # encoding: data bao gồm emoji, kí tự đặc biệt,...
    f = open(filename, "r", encoding='latin-1')
    file = f.readlines()
    label, data = [], []
    A = []
    for line in file:
        thisline = line.strip()
        line_content = thisline.split(" ", 1)
        A.append([int(line_content[0]), ViTokenizer.tokenize(line_content[1])])
    f.close()
    return A

if __name__ == '__main__':
    filename = "train.txt"
    # load data
    A = load_data(filename)
    B = np.array(A)
    train, test = train_test_split(B, test_size=0.2)
    train = np.array(train)
    test = np.array(test)
    # print(A)

    # PRECESSOR:
    x_train, y_train = train[:, 1], train[:, 0]
    x_test, y_test = test[:, 1], test[:, 0]

    # create a transform
    vectorizer = TfidfVectorizer(use_idf=True)

    # tokenize and build vocab
    X = vectorizer.fit_transform(x_train)
    print(vectorizer.vocabulary_)

    # encode document
    vec = vectorizer.transform(x_test).toarray()

    # TRAINING:
    # create model training
    mnb = MultinomialNB()
    KNN = KNeighborsClassifier(n_neighbors=2)
    BNB = BernoulliNB()
    SVC = SVC()
    LSVC = LinearSVC()

    # TRAIN AND TEST
    mnb.fit(X, y_train)
    y_pred = mnb.predict(vec)
    print("Accuracy MultinomialNB: {} %".format(accuracy_score(y_test, y_pred) * 100))

    BNB.fit(X, y_train)
    y_pred = BNB.predict(vec)
    print("Accuracy BernoulliNB: {} %".format(accuracy_score(y_test, y_pred) * 100))

    SVC.fit(X, y_train)
    y_pred = SVC.predict(vec)
    print("Accuracy SVC: {} %".format(accuracy_score(y_test, y_pred) * 100))

    LSVC.fit(X, y_train)
    y_pred = LSVC.predict(vec)
    print("Accuracy LSVC: {} %".format(accuracy_score(y_test, y_pred) * 100))

    KNN.fit(X, y_train)
    y_pred = KNN.predict(vec)
    print("Accuracy KNN: {} %".format(accuracy_score(y_test, y_pred) * 100))
