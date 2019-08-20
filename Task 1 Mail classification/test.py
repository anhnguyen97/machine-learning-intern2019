import json
import pickle

import numpy as np
from DataLoader import *
from feature_extraction import *
from nlp import *


class Test(object):
    def __init__(self, modelPath, dataPath, vectPath):
        self.modelPath = modelPath
        self.dataPath = dataPath
        self.vectPath = vectPath

    def load_model(self):
        return pickle.load(open(self.modelPath, "rb"))

    def save_model(self, model, path):
        pickle.dump(model, open(path, 'wb'))

    def test(self):
        # get test data
        data_loader = DataLoader(self.dataPath[0])
        content = data_loader.read_file()
        X_test1, y_test1 = data_loader.get_data(content)
        data_loader = DataLoader(self.dataPath[1])
        content = data_loader.read_file()
        X_test2, y_test2 = data_loader.get_data(content)
        X_test_init = X_test1 + X_test2
        y_test = y_test1 + y_test2
        # print(X_test)

        # transform test data
        nlp = NLP()
        X_test = nlp.preprocessing(X_test_init)

        feature_extraction = FeatureExtraction()
        transform = feature_extraction.load_vect(self.vectPath)
        X_test = transform.transform(X_test)

        # load model
        loaded_model = self.load_model()
        predict = loaded_model.predict(X_test)
        result = loaded_model.score(X_test, y_test)
        model_name = modelPath.split("/")[2].split(".")[0]
        print("Accuracy - {} : {} %".format(model_name, result * 100))

        result = {
            "model_name": model_name,
            "result": result,
            "predict": predict,
            "X_test": X_test_init,
            'model': loaded_model,
        }

        return result


if __name__ == '__main__':
    dataPath = ["./data/test/ham_test.txt", "./data/test/spam_test.txt"]

    model = []

    print("Using count_vect: ")
    modelPath = "./model/BernoulliNB-countVt.pkl"
    vectPath = "./vect/count_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    model.append(test.test())

    modelPath = "./model/LinearSVC-countVt.pkl"
    vectPath = "./vect/count_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    model.append(test.test())

    modelPath = "./model/MultinomialNB-countVt.pkl"
    vectPath = "./vect/count_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    model.append(test.test())

    print("Using tfidf: ")
    modelPath = "./model/BernoulliNB-wordTfIdf.pkl"
    vectPath = "./vect/word_tfidf_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    model.append(test.test())

    modelPath = "./model/LinearSVC-wordTfIdf.pkl"
    vectPath = "./vect/word_tfidf_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    model.append(test.test())

    modelPath = "./model/MultinomialNB-wordTfIdf.pkl"
    vectPath = "./vect/word_tfidf_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    model.append(test.test())

    # find best model
    result_list = [i['result'] for i in model]
    index = np.argmax(result_list)
    best_model = model[index]

    # save result best model
    data = {
        "Model": best_model["model_name"],
        "Accuracy": best_model["result"],
    }
    with open("./result/best_model.json", "w") as f:
        f.write(json.dumps(data))

    path = "./result/best_model.pkl"
    test.save_model(best_model["model"], path)


    # save classification result:
    best_predict = (best_model["predict"])
    X_test = best_model['X_test']
    spam_index = [i for i, j in enumerate(best_predict) if j == '-1']
    ham_index = [i for i, j in enumerate(best_predict) if j == '1']

    spam_list = [X_test[i] for i in spam_index]
    with open("./result/test-set/spam_clf.txt", "w") as f:
        for item in spam_list:
            f.write("-1 %s\n" % item)
    f.close()

    ham_list = [X_test[i] for i in ham_index]
    with open("./result/test-set/ham_clf.txt", "w") as f:
        for item in ham_list:
            f.write("1 %s\n" % item)
    f.close()

    print(spam_index)
