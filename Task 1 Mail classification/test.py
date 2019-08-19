import pickle

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

    def test(self):
        # get test data
        data_loader = DataLoader(self.dataPath)
        content = data_loader.read_file()
        X_test, y_test = data_loader.get_data(content)
        # print(X_test)

        # transform test data
        nlp = NLP()
        X_test = nlp.preprocessing(X_test)

        feature_extraction = FeatureExtraction()
        transform = feature_extraction.load_vect(self.vectPath)
        X_test = transform.transform(X_test)

        # load model
        loaded_model = self.load_model()

        result = loaded_model.score(X_test, y_test)
        model_name = modelPath.split("/")[2].split("-")[0]
        print("Accuracy - {} : {} %".format(model_name, result * 100))


if __name__ == '__main__':
    dataPath = "./data/test/spam_test.txt"

    print("Using count_vect: ")
    modelPath = "./model/BernoulliNB-countVt.pkl"
    vectPath = "./vect/count_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    test.test()

    modelPath = "./model/LinearSVC-countVt.pkl"
    vectPath = "./vect/count_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    test.test()

    modelPath = "./model/MultinomialNB-countVt.pkl"
    vectPath = "./vect/count_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    test.test()

    print("Using tfidf: ")
    modelPath = "./model/BernoulliNB-wordTfIdf.pkl"
    vectPath = "./vect/word_tfidf_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    test.test()

    modelPath = "./model/LinearSVC-wordTfIdf.pkl"
    vectPath = "./vect/word_tfidf_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    test.test()

    modelPath = "./model/MultinomialNB-wordTfIdf.pkl"
    vectPath = "./vect/word_tfidf_vect.pkl"
    test = Test(modelPath, dataPath, vectPath)
    test.test()
