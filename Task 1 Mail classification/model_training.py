import pickle

from DataLoader import *
from feature_extraction import *
from nlp import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC

class Classifier(object):

    def __init__(self, model, model_name, filePath):
        self.model = model
        self.filePath = filePath
        self.model_name = model_name

    def save_model(self):
        pickle.dump(self.model, open(self.filePath, 'wb'))

    def model_training(self, X_train, y_train):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
        self.model.fit(X_train, y_train)

        # save model
        self.save_model()

        val_predictions = self.model.predict(X_val)
        print("Accuracy valuation - {}: {} %".format(self.model_name, accuracy_score(val_predictions, y_val) * 100))


if __name__ == '__main__':
    # get training data
    data_loader = DataLoader('./data/train/train.txt', 'latin-1')
    file_content = data_loader.read_file()
    X_train_raw, y_train_raw = data_loader.get_data(file_content)

    # preprocessing training data
    nlp = NLP()
    X_train_preprocessed = nlp.preprocessing(X_train_raw)

    # transform text data to vector
    transform = FeatureExtraction(X_train_preprocessed)

    print("1. Using count vectorizer: ")
    X_train_vector = transform.count_vect()

    X_train, y_train = X_train_vector, y_train_raw
    # traning model
    model = MultinomialNB()
    filePath = "./model/MultinomialNB-countVt.pkl"
    classifier = Classifier(model, 'Multinomial', filePath)
    classifier.model_training(X_train, y_train)

    model = BernoulliNB()
    filePath = "./model/BernoulliNB-countVt.pkl"
    classifier = Classifier(model, 'BernoulliNB', filePath)
    classifier.model_training(X_train, y_train)

    model = LinearSVC()
    filePath = "./model/LinearSVC-countVt.pkl"
    classifier = Classifier(model, 'LinearSVC', filePath)
    classifier.model_training(X_train, y_train)

    # transform text data to vector
    print("2. Using tf-idf: ")
    X_train_vector = transform.word_tfidf_vect()

    X_train, y_train = X_train_vector, y_train_raw

    # traning model
    model = MultinomialNB()
    filePath = "./model/MultinomialNB-wordTfIdf.pkl"
    classifier = Classifier(model, 'MultinomialNB', filePath)
    classifier.model_training(X_train, y_train)

    model = BernoulliNB()
    filePath = "./model/BernoulliNB-wordTfIdf.pkl"
    classifier = Classifier(model, 'BernoulliNB', filePath)
    classifier.model_training(X_train, y_train)

    model = LinearSVC()
    filePath = "./model/LinearSVC-wordTfIdf.pkl"
    classifier = Classifier(model, 'LinearSVC', filePath)
    classifier.model_training(X_train, y_train)
