import pickle

import model_function
from mail_classification_training import *

def test(filename, x_test, y_test):

    # loaded_model = model_function.Model.load(open(filename, "rb")
    loaded_model = pickle.load(open(filename, "rb"))
    # print(loaded_model.vocabulary_)
    result = loaded_model.score(x_test, y_test)
    print("Accuracy - {} : {} %".format(filename, result*100))

if __name__ == '__main__':
    test_file = "./testdata/ham_test.txt"
    ddd = MailClassificationPredict()
    x_train, x_test, y_train, y_test = ddd.get_train_test_data(test_file)
    test("NaiveBayesModel", x_train, y_train)
