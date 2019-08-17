import sys

sys.path.insert(1, './')
import numpy as np
import model_function
from sklearn.model_selection import train_test_split
from model import NaiveBayesModel


class MailClassificationPredict():

    def __init__(self):
        self.test = None

    def load_data(self, filename):
        # encoding: data bao gồm emoji, kí tự đặc biệt,...
        f = open(filename, "r", encoding='latin-1')
        file = f.readlines()
        A = []
        for line in file:
            thisline = line.strip()
            line_content = thisline.split(" ", 1)
            if len(line_content[0])>2:
                line_content=thisline.split("\t", 1)
            A.append([int(line_content[0]), line_content[1]])
        f.close()
        return A

    def get_train_test_data(self, filename = "train.txt"):
        # filename = "train.txt"
        # load data
        A = self.load_data(filename)
        # feature_tranform = processing.FeatureTransformer()
        # A = feature_tranform.transform(A)
        B = np.array(A)

        # Split data into test - train:
        train, test = train_test_split(B, test_size=0.2)
        train = np.array(train)
        test = np.array(test)

        x_train, y_train = train[:, 1], train[:, 0]
        x_test, y_test = test[:, 1], test[:, 0]

        return x_train, x_test, y_train, y_test

    def training(self):
        x_train, x_test, y_train, y_test = self.get_train_test_data()
        model = NaiveBayesModel.NaiveBayesModel()
        clf = model.clf_training.fit(x_train, y_train)
        # model_function = model_function.Model()
        model_function.Model.save(clf, "NaiveBayesModel")


if __name__ == '__main__':
    clf = MailClassificationPredict()
    clf.training()
